use std::env;
use chrono::Local;

use tokio::fs::File; // For asynchronous file operations.
use tokio::io::{self, AsyncReadExt, AsyncSeekExt, AsyncWriteExt}; // For asynchronous I/O traits.

use hamming::distance;
use distances::vectors::cosine;

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

fn process_chunk(chunk1: &[u8], chunk2: &[u8]) -> (Vec<u32>, Vec<f32>) {
    let mut hamming_distances = Vec::new();
    let mut cosine_similarities = Vec::new();
    
    let num_pages = chunk1.len() / PAGE_SIZE;
    for i in 0..num_pages {
        let start = i * PAGE_SIZE;
        let end = start + PAGE_SIZE;
        let page1 = &chunk1[start..end];
        let page2 = &chunk2[start..end];
        
        // Hamming distance
        let hamming_distance = distance(page1, page2) as u32;
        hamming_distances.push(hamming_distance);

        // Cosine similarity
        let page1_f32: Vec<f32> = page1.iter().map(|&x| x as f32).collect();
        let page2_f32: Vec<f32> = page2.iter().map(|&x| x as f32).collect();
        let cosine_similarity = cosine(&page1_f32, &page2_f32);
        cosine_similarities.push(cosine_similarity);
    }

    (hamming_distances, cosine_similarities)
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

    let file1 = Arc::new(Mutex::new(File::open(file1_path).await?));
    let file2 = Arc::new(Mutex::new(File::open(file2_path).await?));
    let hamming_result_file = Arc::new(Mutex::new(File::create(hamming_result_file_path).await?));
    let cosine_result_file = Arc::new(Mutex::new(File::create(cosine_result_file_path).await?));

    // Calculate the total size of the files
    let file1_size = file1.lock().unwrap().metadata().await?.len();
    let file2_size = file2.lock().unwrap().metadata().await?.len();
    
    assert_eq!(file1_size, file2_size, "Files should be of the same size");

    // Calculate the segment size for each thread
    let segment_size = file1_size / THREAD_COUNT as u64;

    let hamming_result_vecs: Arc<Mutex<Vec<Vec<u32>>>> = Arc::new(Mutex::new(vec![Vec::new(); THREAD_COUNT]));
    let cosine_result_vecs: Arc<Mutex<Vec<Vec<f32>>>> = Arc::new(Mutex::new(vec![Vec::new(); THREAD_COUNT]));

    // Spawn multiple threads for parallel processing using Rayon
    (0..THREAD_COUNT).into_par_iter().for_each(|thread_id| {
        let file1 = Arc::clone(&file1);
        let file2 = Arc::clone(&file2);
        let hamming_result_vecs = Arc::clone(&hamming_result_vecs);
        let cosine_result_vecs = Arc::clone(&cosine_result_vecs);

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async move {
            let start_offset = thread_id as u64 * segment_size;
            let end_offset = if thread_id == THREAD_COUNT - 1 {
                file1_size // Last thread processes the remaining part
            } else {
                start_offset + segment_size
            };

            let mut offset = start_offset;
            let mut local_hamming_results = Vec::new(); // Local vector for the thread to reduce contention
            let mut local_cosine_results = Vec::new(); // Local vector for the thread to reduce contention            
            
            while offset < end_offset {
                let chunk1 = read_chunk(&mut file1.lock().unwrap(), offset).await.unwrap_or_else(|_| vec![]);
                let chunk2 = read_chunk(&mut file2.lock().unwrap(), offset).await.unwrap_or_else(|_| vec![]);

                if chunk1.is_empty() || chunk2.is_empty() {
                    break; // Exit loop if either file has no more data
                }

                let (hamming_distances, cosine_similarities) = process_chunk(&chunk1, &chunk2);
                local_hamming_results.extend(hamming_distances);
                local_cosine_results.extend(cosine_similarities);

                offset += CHUNK_SIZE  as u64;
            }

            // Write local results to shared vectors
            hamming_result_vecs.lock().unwrap()[thread_id] = local_hamming_results;
            cosine_result_vecs.lock().unwrap()[thread_id] = local_cosine_results;
        });
    });

    // Write the accumulated results to the output files
    let mut hamming_result_file = hamming_result_file.lock().unwrap();
    let mut cosine_result_file = cosine_result_file.lock().unwrap();

    let hamming_result_vecs = Arc::try_unwrap(hamming_result_vecs).unwrap().into_inner().unwrap();
    let cosine_result_vecs = Arc::try_unwrap(cosine_result_vecs).unwrap().into_inner().unwrap();

    let mut hamming_buffer = String::new();
    let mut cosine_buffer = String::new();

    for result_vec in hamming_result_vecs {
        for &distance in result_vec.iter() {
            hamming_buffer.push_str(&format!("{}\n", distance));
        }
    }
    for result_vec in cosine_result_vecs {
        for &similarity in result_vec.iter() {
            cosine_buffer.push_str(&format!("{}\n", similarity));
        }
    }

    hamming_result_file.write_all(hamming_buffer.as_bytes()).await?;
    cosine_result_file.write_all(cosine_buffer.as_bytes()).await?;

    Ok(())
}
