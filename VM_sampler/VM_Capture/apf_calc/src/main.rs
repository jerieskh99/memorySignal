//! apf_calc -- active-page-fraction (APF) calculator (Plan 05 Wave 4).
//!
//! A lean, dependency-free sibling of `live_delta_calc`. Same CLI contract so the
//! consumer can route to either binary:
//!
//!     apf_calc <prev_image> <curr_image> <output_dir>
//!
//! Reads two raw guest-memory dumps and computes the fraction of 4096-byte pages
//! that differ (a page differs if ANY byte differs), then writes that single APF
//! value to:
//!
//!     <output_dir>/apf/apf_results_par-<epoch_ms>.txt
//!
//! The math matches `plan02_apf_helper._compute_active_page_fraction` EXACTLY:
//!   * different file sizes, or an empty file        -> 0.0
//!   * n_pages = floor(size / 4096); if 0            -> 0.0
//!   * APF = (pages where any byte differs) / n_pages
//! Both sides count integers and divide, so the f64 result is bit-identical.

use std::env;
use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::time::{SystemTime, UNIX_EPOCH};

const PAGE_SIZE: usize = 4096;
const CHUNK_PAGES: usize = 256; // 1 MiB streaming buffer per file

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: {} <prev_image> <curr_image> <output_dir>", args[0]);
        std::process::exit(1);
    }
    let (prev, curr, outdir) = (&args[1], &args[2], &args[3]);

    let apf = match compute_apf(prev, curr) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("apf_calc: error reading dumps: {}", e);
            std::process::exit(1);
        }
    };

    let apf_dir = format!("{}/apf", outdir);
    if let Err(e) = fs::create_dir_all(&apf_dir) {
        eprintln!("apf_calc: cannot create {}: {}", apf_dir, e);
        std::process::exit(1);
    }
    let ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let path = format!("{}/apf_results_par-{}.txt", apf_dir, ms);
    match File::create(&path) {
        Ok(mut f) => {
            if let Err(e) = writeln!(f, "{}", apf) {
                eprintln!("apf_calc: cannot write {}: {}", path, e);
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("apf_calc: cannot create {}: {}", path, e);
            std::process::exit(1);
        }
    }
}

/// APF = fraction of 4096-byte pages that differ between `prev` and `curr`.
/// Mirrors plan02_apf_helper._compute_active_page_fraction.
fn compute_apf(prev: &str, curr: &str) -> std::io::Result<f64> {
    let s1 = fs::metadata(prev)?.len();
    let s2 = fs::metadata(curr)?.len();
    // Different sizes (Python: shape mismatch) or empty -> 0.0
    if s1 != s2 || s1 == 0 {
        return Ok(0.0);
    }
    let n_pages = (s1 as usize) / PAGE_SIZE;
    if n_pages == 0 {
        return Ok(0.0);
    }
    let mut r1 = BufReader::with_capacity(PAGE_SIZE * CHUNK_PAGES, File::open(prev)?);
    let mut r2 = BufReader::with_capacity(PAGE_SIZE * CHUNK_PAGES, File::open(curr)?);
    let mut b1 = vec![0u8; PAGE_SIZE * CHUNK_PAGES];
    let mut b2 = vec![0u8; PAGE_SIZE * CHUNK_PAGES];

    let mut done = 0usize;
    let mut differ = 0usize;
    while done < n_pages {
        let pages = std::cmp::min(CHUNK_PAGES, n_pages - done);
        let want = pages * PAGE_SIZE;
        read_exact(&mut r1, &mut b1[..want])?;
        read_exact(&mut r2, &mut b2[..want])?;
        for p in 0..pages {
            let s = p * PAGE_SIZE;
            if b1[s..s + PAGE_SIZE] != b2[s..s + PAGE_SIZE] {
                differ += 1;
            }
        }
        done += pages;
    }
    Ok(differ as f64 / n_pages as f64)
}

/// Read exactly `buf.len()` bytes; BufReader::read can return short.
fn read_exact<R: Read>(r: &mut R, buf: &mut [u8]) -> std::io::Result<()> {
    let mut off = 0;
    while off < buf.len() {
        let n = r.read(&mut buf[off..])?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "unexpected EOF while reading dump pages",
            ));
        }
        off += n;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_pages(path: &std::path::Path, pages: &[[u8; PAGE_SIZE]]) {
        let mut f = File::create(path).unwrap();
        for pg in pages {
            f.write_all(pg).unwrap();
        }
    }

    #[test]
    fn three_of_ten_pages_differ_is_0_3() {
        let dir = std::env::temp_dir();
        let a = dir.join("apf_calc_test_a.raw");
        let b = dir.join("apf_calc_test_b.raw");
        let pa = vec![[0u8; PAGE_SIZE]; 10];
        let mut pb = pa.clone();
        // flip one byte in pages 0, 4, 9 -> 3 of 10 pages differ
        pb[0][0] = 1;
        pb[4][100] = 9;
        pb[9][PAGE_SIZE - 1] = 255;
        write_pages(&a, &pa);
        write_pages(&b, &pb);
        let apf = compute_apf(a.to_str().unwrap(), b.to_str().unwrap()).unwrap();
        assert_eq!(apf, 0.3);
        let _ = fs::remove_file(&a);
        let _ = fs::remove_file(&b);
    }

    #[test]
    fn identical_is_zero() {
        let dir = std::env::temp_dir();
        let a = dir.join("apf_calc_test_id_a.raw");
        let b = dir.join("apf_calc_test_id_b.raw");
        let pa = vec![[7u8; PAGE_SIZE]; 5];
        write_pages(&a, &pa);
        write_pages(&b, &pa);
        assert_eq!(compute_apf(a.to_str().unwrap(), b.to_str().unwrap()).unwrap(), 0.0);
        let _ = fs::remove_file(&a);
        let _ = fs::remove_file(&b);
    }

    #[test]
    fn size_mismatch_is_zero() {
        let dir = std::env::temp_dir();
        let a = dir.join("apf_calc_test_sz_a.raw");
        let b = dir.join("apf_calc_test_sz_b.raw");
        write_pages(&a, &vec![[0u8; PAGE_SIZE]; 3]);
        write_pages(&b, &vec![[0u8; PAGE_SIZE]; 4]);
        assert_eq!(compute_apf(a.to_str().unwrap(), b.to_str().unwrap()).unwrap(), 0.0);
        let _ = fs::remove_file(&a);
        let _ = fs::remove_file(&b);
    }
}
