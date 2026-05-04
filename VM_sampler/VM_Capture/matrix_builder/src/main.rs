/// matrix_builder
///
/// Reads paired hamming/*.txt and cosine/*.txt delta frames for one test,
/// combines them as D = hamming * exp(j * 2π * cosine) per page element,
/// and writes a NumPy .npy file with dtype complex128 in (pages, frames) layout.
///
/// Usage:
///   matrix_builder \
///     --hamming <hamming_dir> \
///     --cosine  <cosine_dir>  \
///     --output  <out.npy>
///
/// File naming convention expected:
///   <timestamp>_<anything>.txt   (the leading timestamp is used for pairing)
///   where <timestamp> is any prefix of digits/colons/hyphens/dots/plus before
///   the first underscore.
///
/// Each .txt file contains one floating-point value per line (one value per page).
/// All files in a directory must have the same number of lines (= num_pages).

use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use rayon::prelude::*;

// ---------------------------------------------------------------------------
// CLI argument parsing (no external crate)
// ---------------------------------------------------------------------------

struct Args {
    hamming_dir: PathBuf,
    cosine_dir: PathBuf,
    output: PathBuf,
    /// Phase scale factor k in D = hamming * exp(j * k * π * cosine).
    /// Default 2.0 → D = hamming * exp(j * 2π * cosine).
    phase_scale: f64,
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = env::args().collect();
    let mut hamming_dir: Option<PathBuf> = None;
    let mut cosine_dir: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut phase_scale: f64 = 2.0;

    let mut i = 1usize;
    while i < raw.len() {
        match raw[i].as_str() {
            "--hamming" => {
                i += 1;
                hamming_dir = Some(PathBuf::from(&raw[i]));
            }
            "--cosine" => {
                i += 1;
                cosine_dir = Some(PathBuf::from(&raw[i]));
            }
            "--output" => {
                i += 1;
                output = Some(PathBuf::from(&raw[i]));
            }
            "--phase-scale" => {
                i += 1;
                phase_scale = raw[i]
                    .parse::<f64>()
                    .map_err(|e| format!("--phase-scale: {e}"))?;
            }
            "--help" | "-h" => {
                eprintln!(
                    "Usage: matrix_builder --hamming <dir> --cosine <dir> --output <out.npy> [--phase-scale 2.0]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("Unknown argument: {other}")),
        }
        i += 1;
    }

    Ok(Args {
        hamming_dir: hamming_dir.ok_or("--hamming is required")?,
        cosine_dir: cosine_dir.ok_or("--cosine is required")?,
        output: output.ok_or("--output is required")?,
        phase_scale,
    })
}

// ---------------------------------------------------------------------------
// Timestamp extraction
//
// Filenames written by `live_delta_calc` look like:
//   memory_dump_hamming_results_par-20260427124713.txt
//   memory_dump_cosine_results_par-20260427124713.txt
//
// The unique pairing key is the digit string that follows "par-".
// We deliberately use rsplit_once so any future prefix changes that still
// keep the "par-<timestamp>" suffix continue to work, and so files that do
// not match the convention are skipped (returning None).
// ---------------------------------------------------------------------------

fn extract_timestamp(path: &Path) -> Option<String> {
    let stem = path.file_stem()?.to_str()?;
    let (_, ts) = stem.rsplit_once("par-")?;
    if ts.is_empty() {
        return None;
    }
    Some(ts.to_owned())
}

// ---------------------------------------------------------------------------
// Discover and pair files
// Returns a Vec of (timestamp_key, hamming_path, cosine_path) sorted by key.
// ---------------------------------------------------------------------------

fn list_txt_files(dir: &Path) -> io::Result<BTreeMap<String, PathBuf>> {
    let mut map = BTreeMap::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let p = entry.path();
        if p.extension().and_then(|e| e.to_str()) == Some("txt") {
            if let Some(ts) = extract_timestamp(&p) {
                map.insert(ts, p);
            }
        }
    }
    Ok(map)
}

fn pair_frames(
    hamming_dir: &Path,
    cosine_dir: &Path,
) -> io::Result<Vec<(String, PathBuf, PathBuf)>> {
    let h_map = list_txt_files(hamming_dir)?;
    let c_map = list_txt_files(cosine_dir)?;

    let mut pairs: Vec<(String, PathBuf, PathBuf)> = Vec::new();
    for (ts, h_path) in &h_map {
        if let Some(c_path) = c_map.get(ts) {
            pairs.push((ts.clone(), h_path.clone(), c_path.clone()));
        }
    }

    // Sort by timestamp string (ISO-8601 sorts lexicographically).
    pairs.sort_by(|a, b| a.0.cmp(&b.0));

    let n_h = h_map.len();
    let n_c = c_map.len();
    let n_paired = pairs.len();
    eprintln!(
        "[matrix_builder] hamming={n_h} cosine={n_c} paired={n_paired}"
    );
    if n_paired == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "No paired hamming/cosine frames found — check directory paths and filename conventions",
        ));
    }

    Ok(pairs)
}

// ---------------------------------------------------------------------------
// Read one text file into a Vec<f64>
// ---------------------------------------------------------------------------

fn read_frame(path: &Path) -> io::Result<Vec<f64>> {
    let f = File::open(path)?;
    let reader = BufReader::new(f);
    let mut values = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let v: f64 = trimmed.parse().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Cannot parse '{}' in {:?}", trimmed, path),
            )
        })?;
        values.push(v);
    }
    Ok(values)
}

// ---------------------------------------------------------------------------
// NumPy .npy writer for complex128 in (pages, frames) layout (C-order).
//
// NPY v1.0 format:
//   magic: \x93NUMPY
//   major: 1, minor: 0
//   HEADER_LEN (u16 LE)
//   header string (ASCII, padded to 64-byte boundary with spaces, ends with \n)
//   raw data: Fortran-order or C-order array of the given dtype
//
// dtype for complex128: '<c16'  (little-endian, 16 bytes per element: re f64 + im f64)
// ---------------------------------------------------------------------------

fn write_npy_complex128(
    path: &Path,
    data: &[f64],   // interleaved (re, im) per element, length = pages * frames * 2
    pages: usize,
    frames: usize,
) -> io::Result<()> {
    // NumPy 1-D shape = (pages, frames) — we store each element as (re f64, im f64).
    let header_str = format!(
        "{{'descr': '<c16', 'fortran_order': False, 'shape': ({pages}, {frames}), }}"
    );

    // The header block must be padded so that (10 + HEADER_LEN) % 64 == 0
    // 10 = 6 (magic+ver) + 2 (HEADER_LEN) + 2 (already in "10 bytes" but actually
    //   \x93NUMPY \x01 \x00  HEADER_LEN(2 bytes) = 10 bytes total prefix)
    let prefix_len = 10usize; // magic(6) + major(1) + minor(1) + header_len(2)
    let raw_len = header_str.len() + 1; // +1 for '\n'
    let total_before_data = prefix_len + raw_len;
    let pad = if total_before_data % 64 == 0 {
        0
    } else {
        64 - total_before_data % 64
    };
    let padded_header = format!("{}{}\n", header_str, " ".repeat(pad));
    let header_len = padded_header.len() as u16;

    let f = File::create(path)?;
    let mut w = BufWriter::new(f);

    // Magic + version
    w.write_all(b"\x93NUMPY\x01\x00")?;
    // HEADER_LEN (little-endian u16)
    w.write_all(&header_len.to_le_bytes())?;
    // Header
    w.write_all(padded_header.as_bytes())?;

    // Data: flat buffer is already interleaved (re, im) f64 pairs in C-order
    // (outer = pages, inner = frames).  Each f64 written as little-endian bytes.
    // NumPy complex128 == two consecutive LE f64 (real then imag) per element.
    for chunk in data.chunks(4096) {
        let bytes: Vec<u8> = chunk.iter().flat_map(|v| v.to_le_bytes()).collect();
        w.write_all(&bytes)?;
    }
    w.flush()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> io::Result<()> {
    let args = parse_args().unwrap_or_else(|e| {
        eprintln!("[matrix_builder] ERROR: {e}");
        eprintln!("Usage: matrix_builder --hamming <dir> --cosine <dir> --output <out.npy> [--phase-scale 2.0]");
        std::process::exit(1);
    });

    eprintln!("[matrix_builder] hamming dir : {:?}", args.hamming_dir);
    eprintln!("[matrix_builder] cosine  dir : {:?}", args.cosine_dir);
    eprintln!("[matrix_builder] output      : {:?}", args.output);
    eprintln!(
        "[matrix_builder] formula     : D = hamming * exp(j * {} * π * cosine)",
        args.phase_scale
    );

    let t0 = Instant::now();

    // --- Discover and pair frames ---
    let pairs = pair_frames(&args.hamming_dir, &args.cosine_dir)?;
    let num_frames = pairs.len();

    // --- Read first frame to learn num_pages ---
    let first_h = read_frame(&pairs[0].1)?;
    let num_pages = first_h.len();
    eprintln!("[matrix_builder] pages={num_pages}  frames={num_frames}");

    // --- Read all frames in parallel ---
    // Each element is a tuple (hamming_vec, cosine_vec) indexed by frame.
    let frame_data: Vec<(Vec<f64>, Vec<f64>)> = pairs
        .par_iter()
        .map(|(ts, h_path, c_path)| {
            let h = read_frame(h_path).unwrap_or_else(|e| {
                panic!("Cannot read hamming frame {ts}: {e}");
            });
            let c = read_frame(c_path).unwrap_or_else(|e| {
                panic!("Cannot read cosine frame {ts}: {e}");
            });
            if h.len() != num_pages {
                panic!(
                    "Frame {ts}: hamming has {} values, expected {num_pages}",
                    h.len()
                );
            }
            if c.len() != num_pages {
                panic!(
                    "Frame {ts}: cosine has {} values, expected {num_pages}",
                    c.len()
                );
            }
            (h, c)
        })
        .collect();

    eprintln!("[matrix_builder] Frames loaded in {:.2?}", t0.elapsed());

    // --- Build interleaved (re, im) buffer in (pages, frames) layout ---
    // Layout: for page p, frame f → index = p * num_frames + f
    // We store (re, im) consecutively → flat buffer length = pages * frames * 2
    let k = args.phase_scale * std::f64::consts::PI;

    // Allocate flat buffer: pages × frames × 2 f64 values
    let total = num_pages * num_frames * 2;
    let mut flat: Vec<f64> = vec![0.0f64; total];

    // Fill: for each page p (outer) and frame f (inner)
    // index into flat = (p * num_frames + f) * 2
    flat.par_chunks_mut(num_frames * 2)
        .enumerate()
        .for_each(|(p, row)| {
            for f in 0..num_frames {
                let h = frame_data[f].0[p];
                let c = frame_data[f].1[p];
                let angle = k * c;
                let re = h * angle.cos();
                let im = h * angle.sin();
                row[f * 2]     = re;
                row[f * 2 + 1] = im;
            }
        });

    eprintln!("[matrix_builder] Complex matrix computed in {:.2?}", t0.elapsed());

    // --- Create output directory if needed ---
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }

    // --- Write .npy ---
    write_npy_complex128(&args.output, &flat, num_pages, num_frames)?;

    let elapsed = t0.elapsed();
    let size_mb = (num_pages * num_frames * 16) as f64 / 1_048_576.0;
    eprintln!(
        "[matrix_builder] Done. Written {:.1} MB to {:?} in {elapsed:.2?}",
        size_mb, args.output
    );

    Ok(())
}
