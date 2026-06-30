use std::env;
use chrono::Local;

use tokio::fs::File; // For asynchronous file operations.
use tokio::io::{self, AsyncReadExt, AsyncSeekExt, AsyncWriteExt}; // For asynchronous I/O traits.

use hamming::distance;
use distances::vectors::cosine;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use std::io::Write; // sync Write for the compressor (tokio's AsyncWriteExt is used elsewhere)

use std::sync::{Arc, Mutex}; // To enable shared access to file handles across threads
use rayon::prelude::*; // For parallel iterator

const CHUNK_SIZE: usize = 262144; // 256KB
const PAGE_SIZE: usize = 4096; // 4KB
const THREAD_COUNT: usize = 16; // Number of threads to be used for parallel processing

// Asynchronously read a chunk from a file at the given offset
async fn read_chunk(file: &mut File, offset: u64) -> io::Result<Vec<u8>> {
    let mut buffer = vec![0; CHUNK_SIZE];
    file.seek(io::SeekFrom::Start(offset)).await?;
    let n = file.read(&mut buffer).await?;
    buffer.truncate(n); // Adjust buffer size to actual bytes read
    Ok(buffer)
}

/// 256-bin byte histogram of a page (counts; sum = page length).
fn hist(page: &[u8]) -> [u32; 256] {
    let mut h = [0u32; 256];
    for &b in page {
        h[b as usize] += 1;
    }
    h
}

/// Total within-page byte gradient energy: sum of |page[j+1] - page[j]|.
/// A roughness proxy -- smooth pages (zeros / text) are low, jagged (random) high.
fn grad_energy(page: &[u8]) -> u64 {
    page.windows(2)
        .map(|w| (w[0] as i16 - w[1] as i16).unsigned_abs() as u64)
        .sum()
}

/// Compressed byte-length of a buffer (zlib/deflate). Basis of the informational metrics.
fn csize(data: &[u8]) -> usize {
    let mut e = ZlibEncoder::new(Vec::new(), Compression::default());
    e.write_all(data).unwrap();
    e.finish().unwrap().len()
}

/// Windowed-entropy "structural entropy" proxy: split the page into 256-byte windows,
/// take each window's byte entropy, return the spread (std-dev) of those window
/// entropies -- i.e. how spatially LUMPY the randomness is (a page with one encrypted
/// region scores high; uniform random or uniform text scores low). A lightweight
/// stand-in for the full wavelet structural entropy.
fn struct_entropy(page: &[u8]) -> f64 {
    let mut es: Vec<f64> = Vec::new();
    for chunk in page.chunks(256) {
        let mut h = [0u32; 256];
        for &b in chunk {
            h[b as usize] += 1;
        }
        let m = chunk.len() as f64;
        let mut e = 0.0;
        for &cnt in h.iter() {
            if cnt > 0 {
                let pr = cnt as f64 / m;
                e -= pr * pr.log2();
            }
        }
        es.push(e);
    }
    if es.is_empty() {
        return 0.0;
    }
    let mean = es.iter().sum::<f64>() / es.len() as f64;
    let var = es.iter().map(|e| (e - mean) * (e - mean)).sum::<f64>() / es.len() as f64;
    var.sqrt()
}

/// Lempel-Ziv (LZ76 / Kaspar-Schuster) complexity: the number of distinct factors in
/// the greedy LZ parsing. A direct algorithmic-complexity measure (no compressor).
fn lz_complexity(s: &[u8]) -> u32 {
    let n = s.len();
    if n == 0 {
        return 0;
    }
    let (mut c, mut l, mut i, mut k, mut k_max) = (1u32, 1usize, 0usize, 1usize, 1usize);
    loop {
        if l + k - 1 >= n {
            if k > 1 {
                c += 1;
            }
            break;
        }
        if s[i + k - 1] == s[l + k - 1] {
            k += 1;
        } else {
            if k > k_max {
                k_max = k;
            }
            i += 1;
            if i == l {
                c += 1;
                l += k_max;
                if l >= n {
                    break;
                }
                i = 0;
                k = 1;
                k_max = 1;
            } else {
                k = 1;
            }
        }
    }
    c
}

/// Per-page feature substrate: one struct per (prev, curr) page pair.
///
/// CONVENTION: every channel is a CHANGE reading -- 0 = no change, larger = more
/// change. Identical pages short-circuit to all-zeros (see `page_metrics`).
/// Extend this family by family -- each new metric is one field here plus a few
/// lines in `page_metrics`. See docs/feature_substrate_spec.{md,pdf} and the
/// progress tracker docs/substrate_progress.html.
#[derive(Clone, Default, Debug)]
struct PageMetrics {
    hamming: u32,   // existing -- Family A / positional: bits flipped (popcount of p XOR q)
    cosine: f32,    // existing -- Family B / structure: cosine DISTANCE (0 = identical)
    // --- Family A: positional group (alternatives to Hamming) ---
    l0: u16,        // byte-Hamming: number of CHANGED bytes (0..=4096)
    l1: u32,        // SAD: sum of absolute byte differences
    l2: f32,        // Euclidean: sqrt(sum of squared byte differences)
    linf: u8,       // Chebyshev: the single biggest byte jump (0..=255)
    mean_abs: f32,  // L1 / 4096: average per-byte change
    gradient_mag: f32, // change in within-page roughness: grad_energy(q) - grad_energy(p)
    // --- Family A: distributional group (byte-histogram distances; 0 = identical histogram) ---
    tv: f32,            // Total Variation: 0.5 * sum |P_b - Q_b|
    chi2: f32,          // symmetric chi-square: sum (P_b - Q_b)^2 / (P_b + Q_b)
    hellinger: f32,     // (1/sqrt2) * sqrt(sum (sqrt(P_b) - sqrt(Q_b))^2)
    kl: f32,            // Kullback-Leibler: sum P_b * log2(P_b / Q_b)  (eps-smoothed)
    js: f32,            // Jensen-Shannon divergence (bounded [0,1])
    wasserstein: f32,   // 1D Wasserstein-1 = sum |cumsum(P) - cumsum(Q)| over bins
    bhattacharyya: f32, // -ln( sum sqrt(P_b * Q_b) )  (eps-floored)
    hist_inter_dist: f32, // 1 - sum min(P_b, Q_b)  (distance form of histogram intersection)
    // --- Family A: informational ---
    ent_delta: f32,         // entropy delta: H(q) - H(p)  (signed; bits/byte)
    csize_delta: f32,       // compressed-size delta: len(C(q)) - len(C(p))  (signed)
    ncd: f32,               // Normalized Compression Distance (0 = identical)
    struct_ent_change: f32, // change in windowed-entropy lumpiness (structural-entropy proxy)
    lz_change: f32,         // LZ76 factor-count change: lz(q) - lz(p)  (signed)
}

/// Compute every per-page metric for one page pair in a single set of passes.
fn page_metrics(p: &[u8], q: &[u8]) -> PageMetrics {
    let hamming = distance(p, q) as u32;
    if hamming == 0 {
        // Identical page = no change. Every channel is 0; skip all the work
        // (unchanged pages are the bulk of a real dump).
        return PageMetrics::default();
    }

    // Family B -- cosine DISTANCE (the page differs here, so norms are well defined
    // unless one side is all-zero, which the `distances` crate maps to 1 = max).
    let p_f32: Vec<f32> = p.iter().map(|&x| x as f32).collect();
    let q_f32: Vec<f32> = q.iter().map(|&x| x as f32).collect();
    let cos = cosine(&p_f32, &q_f32);

    // Family A -- positional group: one byte-wise pass.
    let mut l0: u16 = 0;
    let mut l1: u64 = 0;
    let mut sse: u64 = 0;
    let mut linf: u8 = 0;
    for (&a, &b) in p.iter().zip(q.iter()) {
        let d = (a as i16 - b as i16).unsigned_abs(); // 0..=255
        if d != 0 {
            l0 += 1;
        }
        l1 += d as u64;
        sse += (d as u64) * (d as u64);
        if d as u8 > linf {
            linf = d as u8;
        }
    }
    let gradient_mag = grad_energy(q) as f32 - grad_energy(p) as f32;

    // Family A -- distributional group + entropy: from the two byte histograms.
    let hp = hist(p);
    let hq = hist(q);
    let n = PAGE_SIZE as f64;
    let eps = 1e-12_f64;

    let mut tv = 0.0;
    let mut chi2 = 0.0;
    let mut hell = 0.0; // sum of (sqrt(p)-sqrt(q))^2, rooted at the end
    let mut kl = 0.0;
    let mut js = 0.0;
    let mut bc = 0.0; // Bhattacharyya coefficient
    let mut inter = 0.0;
    let mut hp_ent = 0.0;
    let mut hq_ent = 0.0;
    let mut cp = 0.0; // running CDFs for Wasserstein
    let mut cq = 0.0;
    let mut wass = 0.0;

    for b in 0..256 {
        let pb = hp[b] as f64 / n;
        let qb = hq[b] as f64 / n;
        let m = 0.5 * (pb + qb);

        tv += (pb - qb).abs();
        if pb + qb > 0.0 {
            chi2 += (pb - qb) * (pb - qb) / (pb + qb);
        }
        let s = pb.sqrt() - qb.sqrt();
        hell += s * s;
        if pb > 0.0 {
            kl += pb * (pb / qb.max(eps)).log2();
            js += 0.5 * pb * (pb / m).log2();
            hp_ent -= pb * pb.log2();
        }
        if qb > 0.0 {
            js += 0.5 * qb * (qb / m).log2();
            hq_ent -= qb * qb.log2();
        }
        bc += (pb * qb).sqrt();
        inter += pb.min(qb);

        cp += pb;
        cq += qb;
        wass += (cp - cq).abs();
    }

    // Family A -- informational (compression-based). Only changed pages reach here.
    let csz_p = csize(p);
    let csz_q = csize(q);
    let mut pq = Vec::with_capacity(p.len() + q.len());
    pq.extend_from_slice(p);
    pq.extend_from_slice(q);
    let csz_pq = csize(&pq);
    let ncd = (csz_pq as f64 - csz_p.min(csz_q) as f64) / csz_p.max(csz_q).max(1) as f64;

    PageMetrics {
        hamming,
        cosine: cos,
        l0,
        l1: l1 as u32,
        l2: (sse as f32).sqrt(),
        linf,
        mean_abs: l1 as f32 / PAGE_SIZE as f32,
        gradient_mag,
        tv: (0.5 * tv) as f32,
        chi2: chi2 as f32,
        hellinger: (hell.sqrt() / std::f64::consts::SQRT_2) as f32,
        kl: kl as f32,
        js: js as f32,
        wasserstein: wass as f32,
        bhattacharyya: (-(bc.max(eps).ln())) as f32,
        hist_inter_dist: (1.0 - inter) as f32,
        ent_delta: (hq_ent - hp_ent) as f32,
        csize_delta: csz_q as f32 - csz_p as f32,
        ncd: ncd as f32,
        struct_ent_change: (struct_entropy(q) - struct_entropy(p)) as f32,
        lz_change: lz_complexity(q) as f32 - lz_complexity(p) as f32,
    }
}

/// One PageMetrics per page in the chunk.
fn process_chunk(chunk1: &[u8], chunk2: &[u8]) -> Vec<PageMetrics> {
    let num_pages = chunk1.len() / PAGE_SIZE;
    (0..num_pages)
        .map(|i| {
            let start = i * PAGE_SIZE;
            let end = start + PAGE_SIZE;
            page_metrics(&chunk1[start..end], &chunk2[start..end])
        })
        .collect()
}

#[tokio::main]
async fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4  {
        eprintln!("Usage: {} <prev_image> <new_image> <output_dir>", args[0]);
        std::process::exit(1);
    }

    let prev_image_path = &args[1];
    let new_image_path = &args[2];
    let output_dir = &args[3];

    let file1_path = prev_image_path;
    let file2_path = new_image_path;

    let timestamp = Local::now().format("%Y%m%d%H%M%S").to_string();

    //let hamming_result_file_path = format!("C:\\Users\\jeries\\Desktop\\thesis\\results\\1\\hamming\\memory_dump_hamming_results_par-{}.txt", timestamp);
    //let cosine_result_file_path = format!("C:\\Users\\jeries\\Desktop\\thesis\\results\\1\\cosine\\memory_dump_cosine_results_par-{}.txt", timestamp);
    let hamming_result_file_path = format!("{}/hamming/memory_dump_hamming_results_par-{}.txt", output_dir, timestamp);
    let cosine_result_file_path = format!("{}/cosine/memory_dump_cosine_results_par-{}.txt", output_dir, timestamp);
    // Combined per-page substrate: one CSV, one row per page, one column per metric.
    let metrics_csv_path = format!("{}/metrics/page_metrics-{}.csv", output_dir, timestamp);

    // Ensure output subdirs exist (additive; harmless if already present).
    for sub in ["hamming", "cosine", "metrics"] {
        let _ = std::fs::create_dir_all(format!("{}/{}", output_dir, sub));
    }

    let file1 = Arc::new(Mutex::new(File::open(file1_path).await?));
    let file2 = Arc::new(Mutex::new(File::open(file2_path).await?));
    let hamming_result_file = Arc::new(Mutex::new(File::create(hamming_result_file_path).await?));
    let cosine_result_file = Arc::new(Mutex::new(File::create(cosine_result_file_path).await?));
    let metrics_result_file = Arc::new(Mutex::new(File::create(metrics_csv_path).await?));

    // Calculate the total size of the files
    let file1_size = file1.lock().unwrap().metadata().await?.len();
    let file2_size = file2.lock().unwrap().metadata().await?.len();

    assert_eq!(file1_size, file2_size, "Files should be of the same size");

    // Calculate the segment size for each thread
    let segment_size = file1_size / THREAD_COUNT as u64;

    let result_vecs: Arc<Mutex<Vec<Vec<PageMetrics>>>> =
        Arc::new(Mutex::new(vec![Vec::new(); THREAD_COUNT]));

    // Spawn multiple threads for parallel processing using Rayon
    (0..THREAD_COUNT).into_par_iter().for_each(|thread_id| {
        let file1 = Arc::clone(&file1);
        let file2 = Arc::clone(&file2);
        let result_vecs = Arc::clone(&result_vecs);

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async move {
            let start_offset = thread_id as u64 * segment_size;
            let end_offset = if thread_id == THREAD_COUNT - 1 {
                file1_size // Last thread processes the remaining part
            } else {
                start_offset + segment_size
            };

            let mut offset = start_offset;
            let mut local_results: Vec<PageMetrics> = Vec::new(); // per-thread, reduces lock contention

            while offset < end_offset {
                let chunk1 = read_chunk(&mut file1.lock().unwrap(), offset).await.unwrap_or_else(|_| vec![]);
                let chunk2 = read_chunk(&mut file2.lock().unwrap(), offset).await.unwrap_or_else(|_| vec![]);

                if chunk1.is_empty() || chunk2.is_empty() {
                    break; // Exit loop if either file has no more data
                }

                local_results.extend(process_chunk(&chunk1, &chunk2));

                offset += CHUNK_SIZE  as u64;
            }

            // Write local results to shared vector
            result_vecs.lock().unwrap()[thread_id] = local_results;
        });
    });

    // Write the accumulated results to the output files
    let mut hamming_result_file = hamming_result_file.lock().unwrap();
    let mut cosine_result_file = cosine_result_file.lock().unwrap();
    let mut metrics_result_file = metrics_result_file.lock().unwrap();

    let result_vecs = Arc::try_unwrap(result_vecs).unwrap().into_inner().unwrap();

    let mut hamming_buffer = String::new();
    let mut cosine_buffer = String::new();
    // Header row for the combined substrate CSV (one column per metric).
    let mut metrics_buffer = String::from(
        "hamming,cosine,l0,l1,l2,linf,mean_abs,gradient_mag,tv,chi2,hellinger,kl,js,wasserstein,bhattacharyya,hist_inter_dist,ent_delta,csize_delta,ncd,struct_ent_change,lz_change\n",
    );

    for result_vec in &result_vecs {
        for m in result_vec.iter() {
            // Backward-compatible single-metric files (unchanged format).
            hamming_buffer.push_str(&format!("{}\n", m.hamming));
            cosine_buffer.push_str(&format!("{}\n", m.cosine));
            // Combined substrate row: all metrics for this page.
            metrics_buffer.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                m.hamming, m.cosine, m.l0, m.l1, m.l2, m.linf, m.mean_abs, m.gradient_mag,
                m.tv, m.chi2, m.hellinger, m.kl, m.js, m.wasserstein, m.bhattacharyya,
                m.hist_inter_dist, m.ent_delta, m.csize_delta, m.ncd, m.struct_ent_change,
                m.lz_change
            ));
        }
    }

    hamming_result_file.write_all(hamming_buffer.as_bytes()).await?;
    cosine_result_file.write_all(cosine_buffer.as_bytes()).await?;
    metrics_result_file.write_all(metrics_buffer.as_bytes()).await?;

    Ok(())
}
