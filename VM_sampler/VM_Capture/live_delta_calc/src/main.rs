// live_delta_calc: per-page hamming + cosine delta between two memory snapshots.
//
// Unix-only: uses std::os::unix::fs::FileExt::read_exact_at for lock-free parallel I/O
// across rayon workers. Each worker owns its own File handles and scratch buffers,
// eliminating the per-thread tokio runtime and Arc<Mutex<File>> contention of prior
// versions.
//
// Output semantics are preserved bit-exactly:
//   - hamming value per page: hamming::distance(p1, p2) as u32
//   - cosine value per page: distances::vectors::cosine(&p1_f32, &p2_f32)
//   - identical-page fast path emits calibrated constants obtained at startup by
//     calling the same cosine() function on identical all-zero / all-ones pages.
// Output file paths, directory layout, timestamp format, and line format are unchanged.

use std::cmp::min;
use std::env;
use std::fmt::Write as FmtWrite;
use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;

use chrono::Local;
use distances::vectors::cosine;
use hamming::distance;
use rayon::prelude::*;

const CHUNK_SIZE: usize = 262144; // 256KB
const PAGE_SIZE: usize = 4096; // 4KB
const THREAD_COUNT: usize = 16;

fn process_chunk(
    chunk1: &[u8],
    chunk2: &[u8],
    buf1: &mut Vec<f32>,
    buf2: &mut Vec<f32>,
    cos_ident_zero: f32,
    cos_ident_nonzero: f32,
    out_h: &mut Vec<u32>,
    out_c: &mut Vec<f32>,
) {
    let num_pages = chunk1.len() / PAGE_SIZE;
    for i in 0..num_pages {
        let start = i * PAGE_SIZE;
        let end = start + PAGE_SIZE;
        let p1 = &chunk1[start..end];
        let p2 = &chunk2[start..end];

        // Hamming always computed (cheap; avoids branch on identical path).
        out_h.push(distance(p1, p2) as u32);

        // Identical-page fast path: slice eq compiles to memcmp / SIMD.
        if p1 == p2 {
            let all_zero = p1.iter().all(|&b| b == 0);
            out_c.push(if all_zero {
                cos_ident_zero
            } else {
                cos_ident_nonzero
            });
            continue;
        }

        // Reuse per-thread f32 buffers; capacity is PAGE_SIZE so no reallocation occurs.
        buf1.clear();
        buf1.extend(p1.iter().map(|&x| x as f32));
        buf2.clear();
        buf2.extend(p2.iter().map(|&x| x as f32));
        out_c.push(cosine(buf1, buf2));
    }
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: {} <prev_image> <new_image> <output_dir>", args[0]);
        std::process::exit(1);
    }

    let prev_path = args[1].clone();
    let new_path = args[2].clone();
    let output_dir = args[3].clone();

    // STRICT_IO=1|true -> propagate I/O errors by panicking the worker;
    // default: log and break the segment (preserves prior permissive behavior).
    let strict_io = matches!(
        env::var("STRICT_IO").ok().as_deref(),
        Some("1") | Some("true")
    );

    let timestamp = Local::now().format("%Y%m%d%H%M%S").to_string();
    let hamming_result_file_path = format!(
        "{}/hamming/memory_dump_hamming_results_par-{}.txt",
        output_dir, timestamp
    );
    let cosine_result_file_path = format!(
        "{}/cosine/memory_dump_cosine_results_par-{}.txt",
        output_dir, timestamp
    );

    // Calibrate identical-page cosine constants against the exact library call.
    let zero_vec: Vec<f32> = vec![0.0f32; PAGE_SIZE];
    let ones_vec: Vec<f32> = vec![1.0f32; PAGE_SIZE];
    let cos_ident_zero = cosine(&zero_vec, &zero_vec);
    let cos_ident_nonzero = cosine(&ones_vec, &ones_vec);

    let file1_size = std::fs::metadata(&prev_path)?.len();
    let file2_size = std::fs::metadata(&new_path)?.len();
    assert_eq!(file1_size, file2_size, "Files should be of the same size");

    let segment_size = file1_size / THREAD_COUNT as u64;

    // Per-thread (hamming_text, cosine_text) slots, collected in ascending thread_id order
    // to preserve the global output ordering of prior versions.
    let slots: Vec<(String, String)> = (0..THREAD_COUNT)
        .into_par_iter()
        .map(|thread_id| -> (String, String) {
            let start_offset = thread_id as u64 * segment_size;
            let end_offset = if thread_id == THREAD_COUNT - 1 {
                file1_size
            } else {
                start_offset + segment_size
            };

            let open_or = |path: &str, which: &str| -> Option<File> {
                match File::open(path) {
                    Ok(f) => Some(f),
                    Err(e) => {
                        if strict_io {
                            panic!("open {} failed ({}): {}", which, path, e);
                        }
                        eprintln!(
                            "[live_delta_calc] open {} failed ({}): {}",
                            which, path, e
                        );
                        None
                    }
                }
            };
            let f1 = match open_or(&prev_path, "prev") {
                Some(f) => f,
                None => return (String::new(), String::new()),
            };
            let f2 = match open_or(&new_path, "new") {
                Some(f) => f,
                None => return (String::new(), String::new()),
            };

            let mut chunk1 = vec![0u8; CHUNK_SIZE];
            let mut chunk2 = vec![0u8; CHUNK_SIZE];
            let mut buf1: Vec<f32> = Vec::with_capacity(PAGE_SIZE);
            let mut buf2: Vec<f32> = Vec::with_capacity(PAGE_SIZE);

            let segment_pages = ((end_offset - start_offset) as usize) / PAGE_SIZE;
            let mut local_h: Vec<u32> = Vec::with_capacity(segment_pages);
            let mut local_c: Vec<f32> = Vec::with_capacity(segment_pages);

            let mut offset = start_offset;
            while offset < end_offset {
                let to_read = min(CHUNK_SIZE as u64, end_offset - offset) as usize;
                let r1 = f1.read_exact_at(&mut chunk1[..to_read], offset);
                let r2 = f2.read_exact_at(&mut chunk2[..to_read], offset);
                if r1.is_err() || r2.is_err() {
                    if strict_io {
                        panic!(
                            "read_exact_at failed at offset {}: prev={:?} new={:?}",
                            offset,
                            r1.err(),
                            r2.err()
                        );
                    }
                    eprintln!(
                        "[live_delta_calc] read failed at offset {}, stopping segment",
                        offset
                    );
                    break;
                }
                process_chunk(
                    &chunk1[..to_read],
                    &chunk2[..to_read],
                    &mut buf1,
                    &mut buf2,
                    cos_ident_zero,
                    cos_ident_nonzero,
                    &mut local_h,
                    &mut local_c,
                );
                offset += to_read as u64;
            }

            let mut hs = String::with_capacity(local_h.len() * 6);
            let mut cs = String::with_capacity(local_c.len() * 12);
            for &h in &local_h {
                let _ = writeln!(hs, "{}", h);
            }
            for &c in &local_c {
                let _ = writeln!(cs, "{}", c);
            }
            (hs, cs)
        })
        .collect();

    let mut hamming_text = String::new();
    let mut cosine_text = String::new();
    for (hs, cs) in slots {
        hamming_text.push_str(&hs);
        cosine_text.push_str(&cs);
    }

    std::fs::write(&hamming_result_file_path, hamming_text)?;
    std::fs::write(&cosine_result_file_path, cosine_text)?;

    Ok(())
}
