//! Thin binary: CLI, async file IO, rayon threading, CSV output.
//! All metric computation lives in the library (`live_delta_calc_modular::metrics`).

use std::env;
use chrono::Local;

use tokio::fs::File;
use tokio::io::{self, AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

use std::sync::{Arc, Mutex};
use rayon::prelude::*;

use live_delta_calc_modular::metrics::{self, PageMetrics};

const CHUNK_SIZE: usize = 262144; // 256KB
const THREAD_COUNT: usize = 16; // Number of threads to be used for parallel processing

// Asynchronously read a chunk from a file at the given offset
async fn read_chunk(file: &mut File, offset: u64) -> io::Result<Vec<u8>> {
    let mut buffer = vec![0; CHUNK_SIZE];
    file.seek(io::SeekFrom::Start(offset)).await?;
    let n = file.read(&mut buffer).await?;
    buffer.truncate(n); // Adjust buffer size to actual bytes read
    Ok(buffer)
}

#[tokio::main]
async fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: {} <prev_image> <new_image> <output_dir>", args[0]);
        std::process::exit(1);
    }

    let prev_image_path = &args[1];
    let new_image_path = &args[2];
    let output_dir = &args[3];

    let file1_path = prev_image_path;
    let file2_path = new_image_path;

    let timestamp = Local::now().format("%Y%m%d%H%M%S").to_string();

    let hamming_result_file_path =
        format!("{}/hamming/memory_dump_hamming_results_par-{}.txt", output_dir, timestamp);
    let cosine_result_file_path =
        format!("{}/cosine/memory_dump_cosine_results_par-{}.txt", output_dir, timestamp);
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

    let file1_size = file1.lock().unwrap().metadata().await?.len();
    let file2_size = file2.lock().unwrap().metadata().await?.len();

    assert_eq!(file1_size, file2_size, "Files should be of the same size");

    let segment_size = file1_size / THREAD_COUNT as u64;

    let result_vecs: Arc<Mutex<Vec<Vec<PageMetrics>>>> =
        Arc::new(Mutex::new(vec![Vec::new(); THREAD_COUNT]));

    (0..THREAD_COUNT).into_par_iter().for_each(|thread_id| {
        let file1 = Arc::clone(&file1);
        let file2 = Arc::clone(&file2);
        let result_vecs = Arc::clone(&result_vecs);

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async move {
            let start_offset = thread_id as u64 * segment_size;
            let end_offset = if thread_id == THREAD_COUNT - 1 {
                file1_size
            } else {
                start_offset + segment_size
            };

            let mut offset = start_offset;
            let mut local_results: Vec<PageMetrics> = Vec::new();

            while offset < end_offset {
                let chunk1 = read_chunk(&mut file1.lock().unwrap(), offset).await.unwrap_or_else(|_| vec![]);
                let chunk2 = read_chunk(&mut file2.lock().unwrap(), offset).await.unwrap_or_else(|_| vec![]);

                if chunk1.is_empty() || chunk2.is_empty() {
                    break;
                }

                local_results.extend(metrics::process_chunk(&chunk1, &chunk2));

                offset += CHUNK_SIZE as u64;
            }

            result_vecs.lock().unwrap()[thread_id] = local_results;
        });
    });

    let mut hamming_result_file = hamming_result_file.lock().unwrap();
    let mut cosine_result_file = cosine_result_file.lock().unwrap();
    let mut metrics_result_file = metrics_result_file.lock().unwrap();

    let result_vecs = Arc::try_unwrap(result_vecs).unwrap().into_inner().unwrap();

    let mut hamming_buffer = String::new();
    let mut cosine_buffer = String::new();
    let mut metrics_buffer = String::from(metrics::csv_header());
    metrics_buffer.push('\n');

    for result_vec in &result_vecs {
        for m in result_vec.iter() {
            hamming_buffer.push_str(&format!("{}\n", m.hamming()));
            cosine_buffer.push_str(&format!("{}\n", m.cosine()));
            metrics_buffer.push_str(&metrics::csv_row(m));
            metrics_buffer.push('\n');
        }
    }

    hamming_result_file.write_all(hamming_buffer.as_bytes()).await?;
    cosine_result_file.write_all(cosine_buffer.as_bytes()).await?;
    metrics_result_file.write_all(metrics_buffer.as_bytes()).await?;

    Ok(())
}
