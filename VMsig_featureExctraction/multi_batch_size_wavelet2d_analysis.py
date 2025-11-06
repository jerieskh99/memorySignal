import subprocess
import os
import csv
import json
import gc
from concurrent.futures import ProcessPoolExecutor
from external_zip import zip_dir
import argparse


def run_wavelet_analysis(batches, page_number, wavelet , scattering=False, parallel = True, 
                         npy_only = True, exp_decay=True, pre_combined=True, 
                         test_folder="fourier", analysis_type="decays_scatter"):
    # Build the command dynamically based on the function inputs
    command = [
        "python3", "./wavelet_analysis_features.py", 
        "--analysis", "wavelet2d",
        "--batches", str(batches),
        "--num-pages", str(page_number),
        "--test-folder", test_folder,
        "--analysis-type", analysis_type,
        "--wavelet", wavelet
    ]

    # Add optional flags
    if exp_decay:
        command.append("--exp-decay")
    if pre_combined:
        command.append("--pre-combined")
    if scattering:
        command.append("--scattering")
    if npy_only:
        command.append("--npy-only")
    if parallel:
        command.append("--parallel")
    
    # Print the command to be executed (for logging purposes)
    print(f"Running command: {' '.join(command)}") 
    
    # Run the command using subprocess
    subprocess.run(command)

    gc.collect()  # Clear memory after processing


def run_wavelet_analysis_wrapper(args):
    """
    Wrapper function to unpack the tuple of arguments for run_wavelet_analysis.
    Args will be a tuple (batches, page_number).
    """
    wavelets = ['db4', 'db6', 'coif5', 'sym4', 'sym8', 'haar', 'bior3.3']
    
    batches, page_number = args

    test_folder = "fourier"
    analysis_type = "decays_parallel"


    for wvlt in wavelets:
        run_wavelet_analysis(batches=batches, page_number=page_number, wavelet=wvlt)

    # Define the root directory for all analyses to be zipped
    base_dir = '/Volumes/Extreme SSD/thesis/runs/mathematical'
    output_dir_root = os.path.join(base_dir, test_folder, analysis_type, f'n{page_number}t{batches}')
    
    # Define the zip file path
    zip_output_path = f"{output_dir_root}.zip"

    # Zip the directory using the imported zip_directory function
    zip_dir(output_dir_root, zip_output_path)


def run_scatter_analysis_wrapper(args):
    batches, page_number = args
    test_folder = "fourier"
    analysis_type = "scatter_parallel"

    run_wavelet_analysis(batches=batches, page_number=page_number, wavelet="", scattering = True)

    # Define the root directory for all analyses to be zipped
    base_dir = '/Volumes/Extreme SSD/thesis/runs/mathematical'
    output_dir_root = os.path.join(base_dir, test_folder, analysis_type, f'n{page_number}t{batches}')
    
    # Define the zip file path
    zip_output_path = f"{output_dir_root}.zip"

    # Zip the directory using the imported zip_directory function
    zip_dir(output_dir_root, zip_output_path)


def main():
    # Define the range of batch sizes and page numbers to explore
    # batch_sizes = [4, 8, 16]  # Example batch sizes
    # page_numbers = [256, 512, 1024]  # Example page numbers
    batch_sizes = [10]  # Example batch sizes
    page_numbers = [8192]  # Example page numbers
    # # Loop through each combination of batch size and page number
    # for batch_size in batch_sizes:
    #     for page_num in page_numbers:
    #         print("handling: batch ", batch_size, " , pages ", page_num )
    #         run_wavelet_analysis(batches=batch_size, page_number=page_num)
    batch_combinations = [(batch_size, page_num) for batch_size in batch_sizes for page_num in page_numbers]

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run wavelet analysis.')
    parser.add_argument('--scatter', action='store_true', help='Enable scattering wavelet analysis.')
    args = parser.parse_args()

    is_scatter = args.scatter

    if is_scatter:
        with ProcessPoolExecutor() as executor:
            executor.map(run_scatter_analysis_wrapper, batch_combinations)
    else:
        # Run in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor() as executor:
            # Map the batch combinations to the run_wavelet_analysis function
            executor.map(run_wavelet_analysis_wrapper, batch_combinations)

            # # Submit tasks and get Future objects
            # futures = [executor.submit(run_wavelet_analysis_wrapper, combo) for combo in batch_combinations]

            # # Optionally, wait for all to complete
            # for future in futures:
            #     future.result()  # This will block until each task is done



if __name__ == "__main__":
    main()