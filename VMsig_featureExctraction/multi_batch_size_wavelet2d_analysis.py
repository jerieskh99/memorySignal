import subprocess
import os
import csv
import json
import gc
from concurrent.futures import ProcessPoolExecutor
from external_zip import zip_dir
import argparse


def run_wavelet_analysis(batches, page_number, wavelet , scattering=False, parallel = True, 
                         npy_only = True, exp_decay=True, pre_combined=True, visualize=True, 
                         test_folder="infected", analysis_type="wavelets"):
    # Build the command dynamically based on the function inputs
    command = [
        "python3", "./wavelet_analysis_features_memmapped.py", 
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
    if visualize:
        command.append("--visualize")
    
    # Print the command to be executed (for logging purposes)
    print(f"Running command: {' '.join(command)}") 
    
    # Run the command using subprocess
    subprocess.run(command)

    gc.collect()  # Clear memory after processing


def run_wavelet_normal_analysis(batches, page_number, wavelet , scattering=False, parallel = True, 
                         npy_only = True, exp_decay=True, pre_combined=True, visualize=True, 
                         test_folder="infected", analysis_type="wavelets"):
    # Build the command dynamically based on the function inputs
    command = [
        "python3", "./wavelet_analysis_features_memmapped.py", 
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
    if visualize:
        command.append("--visualize")
    
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
    # wavelets = ['db4', 'db6', 'coif5', 'sym4', 'sym8', 'haar', 'bior3.3']
    wavelets = ['haar']
    
    batches, page_number = args

    test_folder = "infected"
    analysis_type = "wavelets"

    # Define the root directory for all analyses to be zipped
    base_dir = '/Volumes/Extreme SSD/thesis/runs/mixed'
    output_dir_root = os.path.join(base_dir, test_folder, analysis_type, f'n{page_number}t{batches}', "data")

    for wvlt in wavelets:
        wvlt_file_path = os.path.join(output_dir_root, f"{wvlt}_wavelet_real.zip")
        print("wvlt_file_path is ", wvlt_file_path)
        if os.path.exists(wvlt_file_path):
            print(f"{wvlt_file_path} Already Exists")
            continue
        else:
            run_wavelet_normal_analysis(batches=batches, page_number=page_number, wavelet=wvlt)

    
    # # Define the zip file path
    # zip_output_path = f"{output_dir_root}.zip"

    # # Zip the directory using the imported zip_directory function
    # zip_dir(output_dir_root, zip_output_path)




def run_wavelet_analysis_wrapper_2(args):
    """
    Wrapper function to unpack the tuple of arguments for run_wavelet_analysis.
    Args will be a tuple (batches, page_number).
    """
    # wavelets = ['db4', 'db6', 'coif5', 'sym4', 'sym8', 'haar', 'bior3.3']
    wavelets = ['sym8']
    
    batches, page_number = args

    test_folder = "infected"
    analysis_type = "wavelets"

    # Define the root directory for all analyses to be zipped
    base_dir = '/Volumes/Extreme SSD/thesis/runs/mixed'
    output_dir_root = os.path.join(base_dir, test_folder, analysis_type, f'n{page_number}t{batches}', "data")

    for wvlt in wavelets:
        wvlt_file_path = os.path.join(output_dir_root, f"{wvlt}_wavelet_real.zip")
        print("wvlt_file_path is ", wvlt_file_path)
        if os.path.exists(wvlt_file_path):
            print(f"{wvlt_file_path} Already Exists")
            continue
        else:
            run_wavelet_normal_analysis(batches=batches, page_number=page_number, wavelet=wvlt)

    
    # # Define the zip file path
    # zip_output_path = f"{output_dir_root}.zip"

    # # Zip the directory using the imported zip_directory function
    # zip_dir(output_dir_root, zip_output_path)




def run_wavelet_analysis_wrapper_3(args):
    """
    Wrapper function to unpack the tuple of arguments for run_wavelet_analysis.
    Args will be a tuple (batches, page_number).
    """
    # wavelets = ['db4', 'db6', 'coif5', 'sym4', 'sym8', 'haar', 'bior3.3']
    wavelets = ['bior3.3']
    
    batches, page_number = args

    test_folder = "infected"
    analysis_type = "wavelets"

    # Define the root directory for all analyses to be zipped
    base_dir = '/Volumes/Extreme SSD/thesis/runs/mixed'
    output_dir_root = os.path.join(base_dir, test_folder, analysis_type, f'n{page_number}t{batches}', "data")

    for wvlt in wavelets:
        wvlt_file_path = os.path.join(output_dir_root, f"{wvlt}_wavelet_real.zip")
        print("wvlt_file_path is ", wvlt_file_path)
        if os.path.exists(wvlt_file_path):
            print(f"{wvlt_file_path} Already Exists")
            continue
        else:
            run_wavelet_normal_analysis(batches=batches, page_number=page_number, wavelet=wvlt)

    
    # # Define the zip file path
    # zip_output_path = f"{output_dir_root}.zip"

    # # Zip the directory using the imported zip_directory function
    # zip_dir(output_dir_root, zip_output_path)



def run_wavelet_analysis_wrapper_4(args):
    """
    Wrapper function to unpack the tuple of arguments for run_wavelet_analysis.
    Args will be a tuple (batches, page_number).
    """
    # wavelets = ['db4', 'db6', 'coif5', 'sym4', 'sym8', 'haar', 'bior3.3']
    wavelets = ['bior3.3']
    
    batches, page_number = args

    test_folder = "infected"
    analysis_type = "wavelets"

    # Define the root directory for all analyses to be zipped
    base_dir = '/Volumes/Extreme SSD/thesis/runs/mixed'
    output_dir_root = os.path.join(base_dir, test_folder, analysis_type, f'n{page_number}t{batches}', "data")

    for wvlt in wavelets:
        wvlt_file_path = os.path.join(output_dir_root, f"{wvlt}_wavelet_real.zip")
        print("wvlt_file_path is ", wvlt_file_path)
        if os.path.exists(wvlt_file_path):
            print(f"{wvlt_file_path} Already Exists")
            continue
        else:
            run_wavelet_normal_analysis(batches=batches, page_number=page_number, wavelet=wvlt)

    
    # # Define the zip file path
    # zip_output_path = f"{output_dir_root}.zip"

    # # Zip the directory using the imported zip_directory function
    # zip_dir(output_dir_root, zip_output_path)




def run_scatter_analysis_wrapper(args):
    batches, page_number = args
    test_folder = "infected"
    analysis_type = "scatter_parallel"

    run_wavelet_analysis(batches=batches, page_number=page_number, wavelet="", scattering = True)

    # Define the root directory for all analyses to be zipped
    base_dir = '/Volumes/Extreme SSD/thesis/runs/mixed'
    output_dir_root = os.path.join(base_dir, test_folder, analysis_type, f'n{page_number}t{batches}')
    
    # Define the zip file path
    zip_output_path = f"{output_dir_root}.zip"

    # Zip the directory using the imported zip_directory function
    zip_dir(output_dir_root, zip_output_path)


def main():
    # Define the range of batch sizes and page numbers to explore
    # batch_sizes = [4, 8, 16]  # Example batch sizes
    # page_numbers = [256, 512, 1024]  # Example page numbers
    batch_sizes = [24]  # Example batch sizes
    page_numbers = [1024]  # Example page numbers
    # # Loop through each combination of batch size and page number
    # for batch_size in batch_sizes:
    #     for page_num in page_numbers:
    #         print("handling: batch ", batch_size, " , pages ", page_num )
    #         run_wavelet_analysis(batches=batch_size, page_number=page_num)
    batch_combinations = [(batch_size, page_num) for batch_size in batch_sizes for page_num in page_numbers]

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run wavelet analysis.')
    parser.add_argument('--scatter', action='store_true', help='Enable scattering wavelet analysis.')
    parser.add_argument('--second-engine', action='store_true', help='Enable second wavelet engine.')
    parser.add_argument('--third-engine', action='store_true', help='Enable third wavelet.')
    parser.add_argument('--fourth-engine', action='store_true', help='Enable third wavelet.')
    parser.add_argument('--sequential',action='store_true', help='Enable third wavelet.')

    args = parser.parse_args()

    is_scatter = args.scatter
    is_second_engine = args.second_engine
    is_third_engine = args.third_engine
    is_fourth_engine = args.fourth_engine

    if is_scatter:
        with ProcessPoolExecutor() as executor:
            executor.map(run_scatter_analysis_wrapper, batch_combinations)
    else:
        # Run in parallel using ProcessPoolExecutor
        if is_second_engine: 
            with ProcessPoolExecutor() as executor:
                executor.map(run_wavelet_analysis_wrapper_2, batch_combinations)
        
        elif is_third_engine:
            with ProcessPoolExecutor() as executor:
                executor.map(run_wavelet_analysis_wrapper_3, batch_combinations)

        elif is_fourth_engine:
            with ProcessPoolExecutor() as executor:
                executor.map(run_wavelet_analysis_wrapper_4, batch_combinations)

        else:
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