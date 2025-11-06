import os
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.stats import entropy
import matplotlib.pyplot as plt
import argparse
import subprocess
from scipy.signal import hilbert
import pywt
import pywt.data
import plotly.graph_objects as go
import shutil
# from kymatio import Scattering1D, Scattering2D
import torch
from kymatio import Scattering1D
from kymatio.torch import Scattering2D  # Switch to PyTorch-based scattering
import dask.array as da
from dask import delayed, compute
from dask.distributed import Client
import threading
from concurrent.futures import ThreadPoolExecutor
import block_feature_extractor
import csv
import json
import h5py 
from concurrent.futures import ProcessPoolExecutor
from external_zip import zip_dir
import subprocess

feature_type_list = ['energy', 'statistical']

# ============================================================
#                       Data Handling
# ============================================================

def save_analysis_data(data, file_path):
    """Saves the analysis data to a .npy file."""
    np.save(file_path, data)


def read_distance_file(file_path):
    """Reads a distance measure file into a numpy array."""
    return np.loadtxt(file_path)


def scatter_flatten_feature_map(tensor_map):
    flattened = {}
    for key, tensor in tensor_map.items():
        # Convert tensor to NumPy array and flatten
        tensor_np = tensor.squeeze().cpu().numpy()  # Remove extra dimensions if necessary
        flattened[key] = tensor_np.flatten().tolist()  # Flatten the array and convert to list for JSON
    return flattened


# def scatter_keep_last_two_dims(tensor_map):
#     flattened = {}
#     for key, tensor in tensor_map.items():
#         # Convert tensor to NumPy array
#         print(f"The shape of the tensor is: {tensor.shape}") 
#         tensor_np = tensor.squeeze().cpu().numpy()  # Remove extra dimensions if necessary

#         # Reshape to flatten everything except the last two dimensions
#         if tensor_np.ndim > 2:  # Ensure we have more than 2 dims to flatten
#             n = tensor_np.shape[:-2]  # Get all dimensions except the last two
#             tensor_np = tensor_np.reshape(-1, tensor_np.shape[-2], tensor_np.shape[-1])  # Flatten all but last two dims

#         flattened[key] = tensor_np.tolist()  # Convert to list for JSON (optional, based on your needs)
    
#     return flattened


def scatter_keep_last_two_dims(tensor_map):
    flattened = {}
    for key, tensor in tensor_map.items():
        # Convert tensor to NumPy array
        print(f"The shape of the tensor is: {tensor.shape}") 
        tensor_np = flatten_block_scattering_coefficients(tensor)

        flattened[key] = tensor_np.numpy()  # Convert to list for JSON (optional, based on your needs)
        print(f"The NEW shape of the tensor is: {tensor_np.shape}") 

    return flattened


def flatten_block_scattering_coefficients(block_tensor):
    """
    Flatten the scattering coefficients for each block.
    
    Args:
    - block_tensor: A tensor representing the scattering coefficients for a block.

    Returns:
    - Flattened 2D representation of the block's scattering coefficients.
    """
    # Step 1: Squeeze the unnecessary dimensions (batch and channel dimensions if applicable)
    block_tensor_squeezed = block_tensor.squeeze()  # Remove dimensions of size 1

    # Step 2: Flatten the scattering coefficients, preserving the 2D image-like structure
    n_samples, height, width = block_tensor_squeezed.shape
    flattened_block = block_tensor_squeezed.reshape(n_samples, -1)  # Flatten to [n_samples, height*width]
    print(f"the shape is {flattened_block.shape}")

    return flattened_block


def scatter_create_csv(flattened, features_dir, effective_p_start ,effective_p_end, t_start, t_end):
    csv_file = os.path.join(features_dir, f'{effective_p_start}_{effective_p_end}_{t_start}_{t_end}_features.csv')
    
    # Open CSV file for writing
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=flattened.keys())
        
        # Write the header
        writer.writeheader()
        
        # Write the row (in this case, it's a single row of features)
        writer.writerow(flattened)
    
    print(f"CSV file saved to {csv_file}")


def scatter_create_json(flattened, features_dir):
    json_file = os.path.join(features_dir, 'features.json')
    
    # Write the flattened dictionary to a JSON file
    with open(json_file, 'w') as file:
        json.dump(flattened, file, indent=4)
    
    print(f"JSON file saved to {json_file}")


def scatter_create_hdf5(flattened, features_dir, effective_p_start ,effective_p_end, t_start, t_end):
    hdf5_file = os.path.join(features_dir, f'{effective_p_start}_{effective_p_end}_{t_start}_{t_end}_features.h5')
    
    # Create and open the HDF5 file
    with h5py.File(hdf5_file, 'w') as h5file:
        # Create datasets from the flattened dictionary
        for key, value in flattened.items():
            h5file.create_dataset(key, data=value)
    
    print(f"HDF5 file saved to {hdf5_file}")



def scatter_create_hdf5_2d(flattened, features_dir, effective_p_start, effective_p_end, t_start, t_end):
    hdf5_file = os.path.join(features_dir, f'{effective_p_start}_{effective_p_end}_{t_start}_{t_end}_features.h5')
    
    # Create and open the HDF5 file
    with h5py.File(hdf5_file, 'w') as h5file:
        # Create datasets from the flattened dictionary
        for key, value in flattened.items():
            # Convert value to NumPy array if it's a list
            value_np = np.array(value)
            h5file.create_dataset(key, data=value_np)
    
    print(f"HDF5 file saved to {hdf5_file}")



def scatter_create_files(tensors, features_dir, effective_p_start ,effective_p_end, t_start, t_end):
    tensor_map = {f'coeffs_{i}': tensor for i, tensor in enumerate(tensors)}
    # flattened = scatter_flatten_feature_map(tensor_map)
    flattened_2d = scatter_keep_last_two_dims(tensor_map)
    # scatter_create_json(flattened, features_dir)
    # scatter_create_hdf5(flattened_2d, features_dir, effective_p_start ,effective_p_end, t_start, t_end)

    scatter_create_hdf5_2d(flattened_2d, features_dir, effective_p_start ,effective_p_end, t_start, t_end)
    # scatter_create_csv(flattened, features_dir, effective_p_start ,effective_p_end, t_start, t_end)
    print(f"Scatter files created in dir: {features_dir}")


def print_custom_help():
    help_message = """
    This script performs various types of analysis on memory distance data.

    Flags:
    --analysis:       Choose the type of analysis to perform. Choices: ['Fourier2d', 'Ceps', 'Fourier1d', 'Hilbert2d', 'wavelet'] (Required)
    --test-folder:    Specify the name of the test folder inside 'results/1/' (Required)
    --analysis-type:  Specify the type of spectrum analysis, e.g., 'KL' (Required)
    --pre-combined:   Use pre-combined data if available. Combine the data in advance and reuse it.
    --help:           Display this help message and examples of usage.
    --combine:        Combine the data from the Hamming and Cosine directories into a single file and save it for later use.

    Example:
    python script.py --analysis Fourier2d --test-folder test1 --analysis-type KL --pre-combined
    """
    print(help_message)


def combine_data(hamming_directory, cosine_directory, combined_data_path, filtered_data_path, bitmap_path):
    # Read all Hamming and Cosine distance files
    hamming_files = [os.path.join(hamming_directory, f) for f in os.listdir(hamming_directory) if f.endswith('.txt')]
    cosine_files = [os.path.join(cosine_directory, f) for f in os.listdir(cosine_directory) if f.endswith('.txt')]

    # Sort files by name to ensure proper time sequence (assuming filenames reflect time order)
    hamming_files.sort()
    cosine_files.sort()

    # Read and stack the distance files as 1D arrays to form 2D arrays
    hamming_data = [read_distance_file(f) for f in hamming_files]
    cosine_data = [read_distance_file(f) for f in cosine_files]

    hamming_stacked = np.column_stack(hamming_data)
    cosine_stacked = np.column_stack(cosine_data)

    # Combine Hamming and Cosine distances into a complex representation
    combined_data = hamming_stacked * np.exp(1j * 2 * np.pi * cosine_stacked)

    # Save combined data file
    np.save(combined_data_path, combined_data)
    print(f"Combined data saved to {combined_data_path}.")

    # Initialize bitmap list
    bitmap = []

    # Filter out rows that are entirely zeros and record their indices
    non_zero_rows = []
    for i, row in enumerate(combined_data):
        if not np.all(row == 0):
            non_zero_rows.append(row)
            bitmap.append(i)

    # Convert the list of non-zero rows back into a numpy array
    non_zero_data = np.array(non_zero_rows)

    # Save the filtered combined data as a separate file
    np.save(filtered_data_path, non_zero_data)
    print(f"Filtered combined data saved to {filtered_data_path}.")

    # Save the bitmap with indices of non-zero rows
    with open(bitmap_path, 'w') as f:
        for line_number in bitmap:
            f.write(f"{line_number}\n")
    print(f"Bitmap saved to {bitmap_path}.")

    return combined_data, non_zero_data


# ============================================================
#                         1D Wavelet
# ============================================================


# Process a single page over time wavelet.
def process_wavelet_1d(signal, wavelet, level=2, is_complex=False):
    """
    Performs a 1D wavelet transform on the given signal and returns the coefficients.

    Parameters:
    signal (np.ndarray): The 1D data array (signal).
    wavelet (str): The wavelet to use for the 1D transform. Default is 'db4'.
    level (int): The level of decomposition for the 1D wavelet transform. Default is 2.
    is_complex (bool): If True, handles the data as complex, as wavelet itself is complex; otherwise processes real and imaginary parts separately.

    Returns:
    list or tuple: The wavelet coefficients for the signal. If is_complex is True, a list of complex coefficients is returned.
                   If is_complex is False, a tuple of lists (real and imaginary coefficients) is returned.
    """
    if is_complex:
        coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
    else:
        real_coeffs = pywt.wavedec(np.real(signal), wavelet=wavelet, level=level)
        imag_coeffs = pywt.wavedec(np.imag(signal), wavelet=wavelet, level=level)

        # Combine the real and imaginary parts
        coeffs = [r + 1j * i for r, i in zip(real_coeffs, imag_coeffs)]
    
    return coeffs


# Create 1d signal/page scalogram
def create_scalogram_1d(wavelet_name, signal, signal_index, signal_plot_dir):
    """
    Creates and saves a scalogram from the 1D signal.

    Parameters:
    signal (np.ndarray): The 1D data array (signal).
    signal_index (int): The index of the current signal being processed.
    output_data_plot_dirs (list): Directories to save the output data and plots.
    is_complex (bool): If True, handles the data as complex; otherwise processes real and imaginary parts separately.
    """
    print(f"Creating scalogram for signal {signal_index}")

    # Set up the parameters for the scalogram
    plt.figure(figsize=(10, 6))
    scales = np.arange(1, 128)

    # Perform CWT on the signal (which is always complex)
    coefficients, frequencies = pywt.cwt(signal, scales, wavelet_name)


    plt.imshow(np.abs(coefficients), extent=[0, signal.size, frequencies[-1], frequencies[0]], cmap='viridis', aspect='auto')
    plt.title(f'Scalogram - Signal {signal_index}')
    plt.xlabel('Time (index)')
    plt.ylabel('Frequency (Hz)')  # Changed to display frequencies
    plt.colorbar(label='Magnitude')


    # Save the scalogram figure
    plt.savefig(os.path.join(signal_plot_dir, f'scalogram_signal_{signal_index}.png'))
    plt.close()

    print(f'Scalogram for Signal {signal_index} processed and images saved.')


def visualize_wavelet_1d(coeffs, signal_index, signal_plot_dir):
    """
    Visualizes and saves the wavelet coefficients for a 1D signal.

    Parameters:
    coeffs (list or tuple): The wavelet coefficients from the 1D wavelet transform.
    signal_index (int): The index of the current signal being processed.
    output_data_plot_dirs (list): Directories to save the output data and plots.
    is_complex (bool): If True, the data is treated as complex; otherwise, the real and imaginary parts are processed separately.
    """

    plt.figure(figsize=(12, 8))

    for i, coeff in enumerate(coeffs):
        real_part = np.real(coeff)
        imag_part = np.imag(coeff)
        magnitude = np.abs(coeff)
        phase = np.angle(coeff)

        plt.subplot(len(coeffs), 1, i + 1)

        # Plot real, imaginary, magnitude, and phase
        plt.plot(real_part, label='Real', color='blue')
        plt.plot(imag_part, label='Imaginary', color='red')
        plt.plot(magnitude, label='Magnitude', color='green', linestyle='--')
        plt.plot(phase, label='Phase', color='orange', linestyle=':')

        plt.title(f'Level {i} - Coefficients (Signal {signal_index})')
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(signal_plot_dir, f'wavelet_1d_signal_{signal_index}.png'))
    plt.close()


# Process a batch of signals/pages over time wavelet.
def process_1d_wavelet(row_initial, combined_data, output_data_plot_dirs, wavelet, level=2, is_complex=False, is_visualize=False):
    """
    Processes each row (corresponding to a signal/page) in the combined data using 1D Wavelet Transform, visualizes and saves the coefficients, and generates scalograms.

    Parameters:
    combined_data (np.ndarray): The 2D array where each row is a 1D signal (e.g., a page over time).
    output_data_plot_dirs (list): Directories to save the output data and plots.
    wavelet (str): The wavelet to use for the 1D transform. Default is 'db4'.
    level (int): The level of decomposition for the 1D wavelet transform. Default is 2.
    is_complex (bool): If True, handles the data as complex; otherwise processes real and imaginary parts separately.
    """

    # Separate the output for data and plots
    output_data_dir, output_plot_dir = output_data_plot_dirs

    num_signals = combined_data.shape[0]

    # Starting point for processing
    start = row_initial if row_initial is not None else 0

    # Define the root directories based on wavelet type
    wavelet_root_data_dir = os.path.join(output_data_dir, f"{wavelet}_wavelet_{'complex' if is_complex else 'real'}")
    wavelet_root_plot_dir = os.path.join(output_plot_dir, f"{wavelet}_wavelet_{'complex' if is_complex else 'real'}")

    # Ensure the directories exist
    os.makedirs(wavelet_root_data_dir, exist_ok=True)
    os.makedirs(wavelet_root_plot_dir, exist_ok=True)

    for signal_index in range(start, num_signals):
        signal = combined_data[signal_index, :]

        # Process the signal using 1D wavelet transform
        coeffs = process_wavelet_1d(signal, wavelet=wavelet, level=level, is_complex=is_complex)

        # Determine the batch segment (1024 signals per batch)
        batch_index = signal_index // 1024
        i_first = batch_index * 1024
        i_last = min((batch_index + 1) * 1024 - 1, num_signals - 1)

        # Create the directory for this batch of signals under the wavelet-specific directory
        batch_dir = os.path.join(wavelet_root_data_dir, f'{wavelet}_1d_signal_{i_first}_{i_last}')
        os.makedirs(batch_dir, exist_ok=True)

        # Create the subdirectory for the current signal index within the batch
        signal_dir = os.path.join(batch_dir, f'signal_index{signal_index}')
        os.makedirs(signal_dir, exist_ok=True)

        # Create directories for storing .npy and .txt files
        npy_dir = os.path.join(signal_dir, 'npy')
        text_dir = os.path.join(signal_dir, 'text')
        os.makedirs(npy_dir, exist_ok=True)
        os.makedirs(text_dir, exist_ok=True)

        # Save the wavelet coefficients for each level (real, imaginary, magnitude, and phase)
        for level_idx, coeff in enumerate(coeffs):
            # Compute real and imaginary parts (if the wavelet is complex, `coeff` will be complex; otherwise, it'll handle both cases)
            real_part = np.real(coeff)
            imag_part = np.imag(coeff)
            magnitude = np.abs(coeff)
            phase = np.angle(coeff)

            # Save real, imaginary, magnitude, and phase coefficients in .npy and .txt formats
            np.save(os.path.join(npy_dir, f'wavelet_real_coeffs_signal_{signal_index}_level_{level_idx}.npy'), real_part)
            np.save(os.path.join(npy_dir, f'wavelet_imag_coeffs_signal_{signal_index}_level_{level_idx}.npy'), imag_part)
            np.save(os.path.join(npy_dir, f'wavelet_magnitude_signal_{signal_index}_level_{level_idx}.npy'), magnitude)
            np.save(os.path.join(npy_dir, f'wavelet_phase_signal_{signal_index}_level_{level_idx}.npy'), phase)

            # Save the same as text files
            np.savetxt(os.path.join(text_dir, f'wavelet_real_coeffs_signal_{signal_index}_level_{level_idx}.txt'), real_part)
            np.savetxt(os.path.join(text_dir, f'wavelet_imag_coeffs_signal_{signal_index}_level_{level_idx}.txt'), imag_part)
            np.savetxt(os.path.join(text_dir, f'wavelet_magnitude_signal_{signal_index}_level_{level_idx}.txt'), magnitude)
            np.savetxt(os.path.join(text_dir, f'wavelet_phase_signal_{signal_index}_level_{level_idx}.txt'), phase)
        
        if (is_visualize) and (signal_index % 1024 == 0):
             # Create a subdirectory in the plot output directory for visualizations
            signal_plot_dir = os.path.join(wavelet_root_plot_dir, f'visualize_1d_signal_{signal_index}')
            os.makedirs(signal_plot_dir, exist_ok=True)

            # Call the existing visualize functions to save the plots
            visualize_wavelet_1d(coeffs, signal_index, signal_plot_dir)
            create_scalogram_1d(wavelet, signal, signal_index, signal_plot_dir)


# Process a batch of signals/pages over time wavelet.
def process_1d_wavelet_time_segment(t_div, row_initial, combined_data, output_data_plot_dirs, wavelet='db4', level=2, is_complex=False, is_visualize=False):
    """
    Processes each row (corresponding to a signal/page) in the combined data using 1D Wavelet Transform, visualizes and saves the coefficients, and generates scalograms.

    Parameters:
    combined_data (np.ndarray): The 2D array where each row is a 1D signal (e.g., a page over time).
    output_data_plot_dirs (list): Directories to save the output data and plots.
    wavelet (str): The wavelet to use for the 1D transform. Default is 'db4'.
    level (int): The level of decomposition for the 1D wavelet transform. Default is 2.
    is_complex (bool): If True, handles the data as complex; otherwise processes real and imaginary parts separately.
    """

    # Separate the output for data and plots
    output_data_dir, output_plot_dir = output_data_plot_dirs

    num_signals, num_samples = combined_data.shape[0], combined_data.shape[1]

    # Size of time/sample batch:
    T = num_samples // t_div

    # Starting point for processing
    start = row_initial if row_initial is not None else 0

    # Define the root directories based on wavelet type
    wavelet_root_data_dir = os.path.join(output_data_dir, f"{wavelet}_wavelet_{'complex' if is_complex else 'real'}")
    wavelet_root_plot_dir = os.path.join(output_plot_dir, f"{wavelet}_wavelet_{'complex' if is_complex else 'real'}")

    # Ensure the directories exist
    os.makedirs(wavelet_root_data_dir, exist_ok=True)
    os.makedirs(wavelet_root_plot_dir, exist_ok=True)

    for batch in range (0, t_div):
        # Define the current batch
        wavelet_batch_data_dir = os.path.join(wavelet_root_data_dir, f'batch_{batch}')
        wavelet_batch_plot_dir = os.path.join(wavelet_root_plot_dir, f'batch_{batch}')

        # Ensure the directories exist
        os.makedirs(wavelet_batch_data_dir, exist_ok=True)
        os.makedirs(wavelet_batch_plot_dir, exist_ok=True)

        # Define the edges of samples of the current batch
        t_start = T * batch
        t_end = t_start + T - 1

        for signal_index in range(start, num_signals):
            signal = combined_data[signal_index, t_start:(t_end+1)]

            # Process the signal using 1D wavelet transform
            coeffs = process_wavelet_1d(signal, wavelet=wavelet, level=level, is_complex=is_complex)

            # Determine the batch segment (1024 signals per batch)
            batch_index = signal_index // 1024
            i_first = batch_index * 1024
            i_last = min((batch_index + 1) * 1024 - 1, num_signals - 1)

            # Create the directory for this batch of signals under the wavelet-specific directory
            batch_dir = os.path.join(wavelet_batch_data_dir, f'{wavelet}_1d_signal_{i_first}_{i_last}')
            os.makedirs(batch_dir, exist_ok=True)

            # Create the subdirectory for the current signal index within the batch
            signal_dir = os.path.join(batch_dir, f'signal_index{signal_index}')
            os.makedirs(signal_dir, exist_ok=True)

            # Create directories for storing .npy and .txt files
            npy_dir = os.path.join(signal_dir, 'npy')
            text_dir = os.path.join(signal_dir, 'text')
            os.makedirs(npy_dir, exist_ok=True)
            os.makedirs(text_dir, exist_ok=True)

            # Save the wavelet coefficients for each level (real, imaginary, magnitude, and phase)
            for level_idx, coeff in enumerate(coeffs):
                # Compute real and imaginary parts (if the wavelet is complex, `coeff` will be complex; otherwise, it'll handle both cases)
                real_part = np.real(coeff)
                imag_part = np.imag(coeff)
                magnitude = np.abs(coeff)
                phase = np.angle(coeff)

                # Save real, imaginary, magnitude, and phase coefficients in .npy and .txt formats
                np.save(os.path.join(npy_dir, f'wavelet_real_coeffs_signal_{signal_index}_level_{level_idx}.npy'), real_part)
                np.save(os.path.join(npy_dir, f'wavelet_imag_coeffs_signal_{signal_index}_level_{level_idx}.npy'), imag_part)
                np.save(os.path.join(npy_dir, f'wavelet_magnitude_signal_{signal_index}_level_{level_idx}.npy'), magnitude)
                np.save(os.path.join(npy_dir, f'wavelet_phase_signal_{signal_index}_level_{level_idx}.npy'), phase)

                # Save the same as text files
                np.savetxt(os.path.join(text_dir, f'wavelet_real_coeffs_signal_{signal_index}_level_{level_idx}.txt'), real_part)
                np.savetxt(os.path.join(text_dir, f'wavelet_imag_coeffs_signal_{signal_index}_level_{level_idx}.txt'), imag_part)
                np.savetxt(os.path.join(text_dir, f'wavelet_magnitude_signal_{signal_index}_level_{level_idx}.txt'), magnitude)
                np.savetxt(os.path.join(text_dir, f'wavelet_phase_signal_{signal_index}_level_{level_idx}.txt'), phase)
            
            if (is_visualize) and (signal_index % 1024 == 0):
                # Create a subdirectory in the plot output directory for visualizations
                signal_plot_dir = os.path.join(wavelet_batch_plot_dir, f'visualize_1d_signal_{signal_index}')
                os.makedirs(signal_plot_dir, exist_ok=True)

                # Call the existing visualize functions to save the plots
                visualize_wavelet_1d(coeffs, signal_index, signal_plot_dir)
                create_scalogram_1d(wavelet, signal, signal_index, signal_plot_dir)
            
            print(f"Processed wavelet for block {signal_index}")


# ============================================================
#                         2D Wavelet
# ============================================================

def flatten_feature_map(feature_map):
    if feature_map is None:
        raise ValueError("The input feature_map is None. Please provide a valid dictionary.")
    
    print("my feature map", feature_map)
    flattened = {}
    for category, levels in feature_map.items():
        for level, features in levels.items():
            for feature_name, value in features.items():
                # Flatten by combining category, level, and feature name
                new_key = f"{category}_{level}_{feature_name}"
                # Check if the value is complex and convert it
                if isinstance(value, complex):
                    flattened[new_key] = {
                        'real': value.real,
                        'imag': value.imag
                    }
                # Check if the value is a NumPy array and convert to list
                elif isinstance(value, np.ndarray):
                    flattened[new_key] = value.tolist()
                # Convert NumPy float or complex to Python float
                elif isinstance(value, (np.float64, np.complex128)):
                    flattened[new_key] = value.item()
                else:
                    flattened[new_key] = value
                    
    return flattened


def create_csv(flattened, features_dir):
    csv_file = os.path.join(features_dir, 'features.csv')
    
    # Open CSV file for writing
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=flattened.keys())
        
        # Write the header
        writer.writeheader()
        
        # Write the row (in this case, it's a single row of features)
        writer.writerow(flattened)
    
    print(f"CSV file saved to {csv_file}")


def create_json(flattened, features_dir):
    json_file = os.path.join(features_dir, 'features.json')
    
    # Write the flattened dictionary to a JSON file
    with open(json_file, 'w') as file:
        json.dump(flattened, file, indent=4)
    
    print(f"JSON file saved to {json_file}")


def create_hdf5(flattened, features_dir):
    hdf5_file = os.path.join(features_dir, 'features.h5')
    
    # Create and open the HDF5 file
    with h5py.File(hdf5_file, 'w') as h5file:
        # Create datasets from the flattened dictionary
        for key, value in flattened.items():
            h5file.create_dataset(key, data=value)
    
    print(f"HDF5 file saved to {hdf5_file}")


def create_files(features, features_dir):
    #Creates features files
    flattened = flatten_feature_map(features)
    create_csv(flattened, features_dir)
    create_json(flattened, features_dir)
    create_hdf5(flattened, features_dir)
    print(f"Files created in dir: {features_dir}")



def process_wavelet_2d(block, wavelet, level=2, is_complex=False, is_mag = False):
    """
    Performs a 2D wavelet transform on the given block of data and returns the coefficients.

    Parameters:
    block (np.ndarray): The 2D data array (block).
    wavelet (str): The wavelet to use for the 2D transform. Default is 'db4'.
    level (int): The level of decomposition for the 2D wavelet transform. Default is 2.
    is_complex (bool): If True, handles the data as complex; otherwise processes real and imaginary parts separately.

    Returns:
    list or tuple: The wavelet coefficients for the block. If is_complex is True, a list of complex coefficients is returned.
                   If is_complex is False, a tuple of lists (real and imaginary coefficients) is returned.
    """
    # Check and print the shape of the input block
    print(f"Block shape: {block.shape}")
    if block.size == 0:
        raise ValueError("Input block is empty! Please provide a non-empty array.")

    if is_complex:
        coeffs = pywt.wavedec2(block, wavelet=wavelet, level=level)
        return coeffs
    else:
        if is_mag:
            coeffs_mag = pywt.wavedec2(np.abs(block), wavelet=wavelet, level=level)
            coeffs_phs = pywt.wavedec2(np.angle(block), wavelet=wavelet, level=level)
        else:
            coeffs_real = pywt.wavedec2(np.real(block), wavelet=wavelet, level=level)
            coeffs_imag = pywt.wavedec2(np.imag(block), wavelet=wavelet, level=level)
        
        # print(len(coeffs_real[1]))
        # print(coeffs_imag[1][1].shape)
        # Construct the complex array using np.complex_
        # complex_array = np.array([np.complex_(r, i) for r, i in zip(real_part, imag_part)])

        #TODO MAKE IT ALSO FOR MAGNITUDE PHASE!!!!!!!!!!!!!!

        # Combine the real and imaginary parts recursively
        if is_mag:
            return (coeffs_mag, coeffs_phs)
        else:
            return (coeffs_real, coeffs_imag)


########################################################
#        Parallel wavelet 2D - features- no decay      #
########################################################

def process_block_2d_wavelet_features_single_block(full_signal, row_initial, n, batch_index, t_start, t_end, num_pages, 
                                                   output_data_plot_dirs, wavelet, level, is_visualize, is_complex):
    start = 0 if row_initial is None else row_initial

    # Separate the output for data and plots
    output_data_dir, output_plot_dir = output_data_plot_dirs

    # Define the root directories based on wavelet type
    wavelet_root_data_dir = os.path.join(output_data_dir, f"{wavelet}_wavelet_{'complex' if is_complex else 'real'}")
    wavelet_root_plot_dir = os.path.join(output_plot_dir, f"{wavelet}_wavelet_{'complex' if is_complex else 'real'}")

    # Ensure the directories exist
    os.makedirs(wavelet_root_data_dir, exist_ok=True)
    os.makedirs(wavelet_root_plot_dir, exist_ok=True)

    # Define the current batch
    wavelet_batch_data_dir = os.path.join(wavelet_root_data_dir, f'batch_{batch_index}')
    wavelet_batch_plot_dir = os.path.join(wavelet_root_plot_dir, f'batch_{batch_index}')

    # Ensure the directories exist
    os.makedirs(wavelet_batch_data_dir, exist_ok=True)
    os.makedirs(wavelet_batch_plot_dir, exist_ok=True)

    for signal_index in range(start, num_pages, n):
        # Define the edges of samples of the current batch
        p_start = signal_index
        p_end = p_start + n - 1
        print('startt ', t_start, ' endt ', t_end )
        print('startp ', p_start, ' endp ', p_end )
        # print('numpages ', num_pages, ' signal_index ', signal_index, ' n ', n )
        # Ensure we don't go out of bounds
        signal = full_signal[p_start:(p_end+1), t_start:(t_end+1)]
        print(f"Processing wavelet for block [{t_start}, {t_end}] X [{p_start}, {p_end}]")
        # Perform 2D wavelet transform on the block
        coeffs = process_wavelet_2d(signal, wavelet=wavelet, level=level, is_complex=is_complex, is_mag=True)

        # Create the directory for this batch of signals under the wavelet-specific directory
        batch_dir = os.path.join(wavelet_batch_data_dir, f'{wavelet}_2d_signal_{p_start}_{p_end}')
        os.makedirs(batch_dir, exist_ok=True)

        features_dir = os.path.join(batch_dir, 'features')
        os.makedirs(features_dir)

        # Visualize
        if is_visualize:
            # Create a subdirectory in the plot output directory for visualizations
            signal_plot_dir = os.path.join(wavelet_batch_plot_dir, f'visualize_1d_signal_{p_start}_{p_end}')
            os.makedirs(signal_plot_dir, exist_ok=True)

            # Call the existing visualize functions to save the plots
            visualize_wavelet_1d(coeffs, signal_index, signal_plot_dir)
            create_scalogram_1d(wavelet, signal, signal_index, signal_plot_dir)

        # Exctract all possible features
        features = block_feature_extractor.process_wavelet_coeffs_block(coeffs, feature_type_list, num_levels=level, coeffs2=None, coeffs_next_batch=None)

        create_files(features, features_dir)

        print(f"Processed wavelet for block [{t_start}, {t_end}] X [{p_start}, {p_end}]")


def process_block_2d_wavelet_features_parallel(full_signal, row_initial, n, num_batches, output_data_plot_dirs, wavelet,
                                              level, is_visualize=False, is_complex=False):
    num_pages, num_samples = full_signal.shape
    batch_size = num_samples // num_batches

    index_map = [(batch, batch * batch_size, (batch + 1) * batch_size - 1) for batch in range(num_batches)]

    with ProcessPoolExecutor(max_workers=num_batches) as executer:
        futures = [
            executer.submit(
                process_block_2d_wavelet_features_single_block, full_signal, row_initial, n, batch_index, 
                t_start_index, t_end_index, num_pages, output_data_plot_dirs, wavelet, level, is_visualize, is_complex
            )
            for batch_index, t_start_index, t_end_index in index_map
        ]

        # Wait for all futures to complete
        for future in futures:
            future.result()
    

########################################################
#             Parallel wavelet 2D - no decay           #
########################################################

# Single block no decay
def process_block_2d_wavelet_features(full_signal, row_initial, n, num_batches, output_data_plot_dirs, 
                                      wavelet, level, is_visualize=False, is_complex=False):
    """
    Processes blocks of the combined data using 2D Wavelet Transform, visualizes and saves the coefficients, and generates scalograms.

    Parameters:
    signal (np.ndarray): The 2D array where rows are pages and columns are time.
    n (int): Number of pages per block.
    t (int): Divisor for the time axis, to determine the width of the block in time.
    output_data_plot_dirs (list): Directories to save the output data and plots.
    wavelet (str): The wavelet to use for the 2D transform. Default is 'db4'.
    level (int): The level of decomposition for the 2D wavelet transform. Default is 2.
    is_complex (bool): If True, handles the data as complex; otherwise processes real and imaginary parts separately.
    is_visualize (bool): If True, visualizes the Scalogram and the Coeffs.
    """

    # Separate the output for data and plots
    output_data_dir, output_plot_dir = output_data_plot_dirs

    num_pages, num_samples = full_signal.shape
    batch_size = num_samples // num_batches

    start = 0 if row_initial is None else row_initial

    # Define the root directories based on wavelet type
    wavelet_root_data_dir = os.path.join(output_data_dir, f"{wavelet}_wavelet_{'complex' if is_complex else 'real'}")
    wavelet_root_plot_dir = os.path.join(output_plot_dir, f"{wavelet}_wavelet_{'complex' if is_complex else 'real'}")

    # Ensure the directories exist
    os.makedirs(wavelet_root_data_dir, exist_ok=True)
    os.makedirs(wavelet_root_plot_dir, exist_ok=True)

    for batch in range (0, num_batches):
        # Define the current batch
        wavelet_batch_data_dir = os.path.join(wavelet_root_data_dir, f'batch_{batch}')
        wavelet_batch_plot_dir = os.path.join(wavelet_root_plot_dir, f'batch_{batch}')

        # Ensure the directories exist
        os.makedirs(wavelet_batch_data_dir, exist_ok=True)
        os.makedirs(wavelet_batch_plot_dir, exist_ok=True)

        # Define the edges of samples of the current batch
        t_start = batch_size * batch
        t_end = t_start + batch_size - 1

        for signal_index in range(start, num_pages, n):
            # Define the edges of samples of the current batch
            p_start = signal_index
            p_end = p_start + n - 1
            print('startt ', t_start, ' endt ', t_end )
            print('startp ', p_start, ' endp ', p_end )
            # print('numpages ', num_pages, ' signal_index ', signal_index, ' n ', n )
            # Ensure we don't go out of bounds
            signal = full_signal[p_start:(p_end+1), t_start:(t_end+1)]
            print(f"Processing wavelet for block [{t_start}, {t_end}] X [{p_start}, {p_end}]")
            # Perform 2D wavelet transform on the block
            coeffs = process_wavelet_2d(signal, wavelet=wavelet, level=level, is_complex=is_complex, is_mag=True)

            # Create the directory for this batch of signals under the wavelet-specific directory
            batch_dir = os.path.join(wavelet_batch_data_dir, f'{wavelet}_2d_signal_{p_start}_{p_end}')
            os.makedirs(batch_dir, exist_ok=True)

            # Create directories for storing .npy and .txt files
            npy_dir = os.path.join(batch_dir, 'npy')
            text_dir = os.path.join(batch_dir, 'text')
            features_dir = os.path.join(batch_dir, 'features')
            os.makedirs(npy_dir, exist_ok=True)
            os.makedirs(text_dir, exist_ok=True)
            os.makedirs(features_dir)

            # Create directories for storing real and image files
            npy_dir_real = os.path.join(npy_dir, 'mag')
            npy_dir_imag = os.path.join(npy_dir, 'phase')
            os.makedirs(npy_dir_real, exist_ok=True)
            os.makedirs(npy_dir_imag, exist_ok=True)

            txt_dir_real = os.path.join(text_dir, 'mag')
            txt_dir_imag = os.path.join(text_dir, 'phase')
            os.makedirs(txt_dir_real, exist_ok=True)
            os.makedirs(txt_dir_imag, exist_ok=True)

            # Visualize
            if is_visualize:
                # Create a subdirectory in the plot output directory for visualizations
                signal_plot_dir = os.path.join(wavelet_batch_plot_dir, f'visualize_1d_signal_{p_start}_{p_end}')
                os.makedirs(signal_plot_dir, exist_ok=True)

                # Call the existing visualize functions to save the plots
                visualize_wavelet_1d(coeffs, signal_index, signal_plot_dir)
                create_scalogram_1d(wavelet, signal, signal_index, signal_plot_dir)

            # Exctract all possible features
            features = block_feature_extractor.process_wavelet_coeffs_block(coeffs, feature_type_list, num_levels=level, coeffs2=None, coeffs_next_batch=None)

            create_files(features, features_dir)

            print(f"Processed wavelet for block [{t_start}, {t_end}] X [{p_start}, {p_end}]")





##################################################
#     Parallel decayed wavelet 2D - Features     #
##################################################

def process_block_2d_wavelet_decay_features_single_batch(full_signal, row_initial, n, batch_index, t_start_idx_flt, t_end_idx__flt, num_pages, output_data_plot_dirs, wavelet, 
                  level, is_visualize, is_complex, page_overlap_percentage, page_decay_rate):
    t_start_idx = int(t_start_idx_flt)
    t_end_idx = int(t_end_idx__flt)
    overlap_pages = int(n * page_overlap_percentage)
    start = 0 if row_initial is None else row_initial

    # Separate the output for data and plots
    output_data_dir, output_plot_dir = output_data_plot_dirs

    # Define the root directories based on wavelet type
    wavelet_root_data_dir = os.path.join(output_data_dir, f"{wavelet}_wavelet_{'complex' if is_complex else 'real'}")
    wavelet_root_plot_dir = os.path.join(output_plot_dir, f"{wavelet}_wavelet_{'complex' if is_complex else 'real'}")
    
    # Ensure the directories exist
    os.makedirs(wavelet_root_data_dir, exist_ok=True)
    os.makedirs(wavelet_root_plot_dir, exist_ok=True)

    # Define the current batch directories
    wavelet_batch_data_dir = os.path.join(wavelet_root_data_dir, f'batch_{batch_index}')
    wavelet_batch_plot_dir = os.path.join(wavelet_root_plot_dir, f'batch_{batch_index}')

    # Ensure the directories exist
    os.makedirs(wavelet_batch_data_dir, exist_ok=True)
    os.makedirs(wavelet_batch_plot_dir, exist_ok=True)

    # Function to wrap around the indices in the time or page axis
    def wrap_index(index, max_value):
        """ Returns a wrapped index to handle negative or out-of-bounds indices. """
        return index % max_value
    
    for page_idx in range(start, num_pages, n):
        # TODO: Check negative modolu
        # Get effective block indices
        effective_p_start = int(wrap_index(int(page_idx - overlap_pages), num_pages))
        effective_p_end   = int(wrap_index(int(effective_p_start + n + overlap_pages), num_pages))

        if effective_p_end > effective_p_start:
            block = full_signal[effective_p_start:effective_p_end, t_start_idx:t_end_idx]
        else:
            print(effective_p_start, " ", num_pages, effective_p_start-num_pages)
            maxpage = int(num_pages)
            block_part1 = full_signal[effective_p_start:maxpage, t_start_idx:t_end_idx]
            
            # Second part from the start of the array to effective_p_end
            block_part2 = full_signal[0:effective_p_end, t_start_idx:t_end_idx]
            
            # Concatenate the two parts to form the full block
            block = np.vstack((block_part1, block_part2))
        
        print("effective_p_start:effective_p_end, t_start_idx:t_end_idx")
        print(effective_p_start, ":", effective_p_end, ", ", t_start_idx, ":", t_end_idx)
        print("block size: ", block.shape, " N:", n)

        single_axis_decay(block, page_decay_rate, n)
        print("decayed dims: ", block.shape)

        # Perform 2D wavelet transform on the block
        coeffs = process_wavelet_2d(block, wavelet=wavelet, level=level, is_complex=is_complex)

        # Create the directory for this batch of signals under the wavelet-specific directory
        batch_dir = os.path.join(wavelet_batch_data_dir, f'{wavelet}_2d_signal_{effective_p_start}_{effective_p_end}')
        os.makedirs(batch_dir, exist_ok=True)

        features_dir = os.path.join(batch_dir, 'features')
        os.makedirs(features_dir)

        if is_visualize:
            # Create a subdirectory in the plot output directory for visualizations
            signal_plot_dir = os.path.join(wavelet_batch_plot_dir, f'visualize_1d_signal_{effective_p_start}_{effective_p_end}')
            os.makedirs(signal_plot_dir, exist_ok=True)

            # Call the existing visualize functions to save the plots
            visualize_wavelet_1d(coeffs, effective_p_start, signal_plot_dir)
            create_scalogram_1d(wavelet, block, effective_p_start, signal_plot_dir)

        # Exctract all possible features
        features = block_feature_extractor.process_wavelet_coeffs_block(coeffs, feature_type_list, num_levels=level, coeffs2=None, coeffs_next_batch=None)

        print(features)
        # import time
        # time.sleep(8)

        create_files(features, features_dir)

        print(f"Processed wavelet for block [{t_start_idx}, {t_end_idx}] X [{effective_p_start}, {effective_p_end}]")
    
    # Zip the batch directory after processing is complete
    zip_output_path = os.path.join(wavelet_root_data_dir, f'batch_{batch_index}.zip')
    zip_dir(wavelet_batch_data_dir, zip_output_path)

    # Remove the original directory after zipping
    # shutil.rmtree(wavelet_batch_data_dir)
    subprocess.run(['rm', '-rf', wavelet_batch_data_dir], check=True)
    print(f"Removed original directory: {wavelet_batch_data_dir}")
    
            



def process_block_2d_wavelet_features_overlap_parallel(full_signal, row_initial, n, num_batches, output_data_plot_dirs, wavelet,
                                              level, is_visualize=False, is_complex=False, page_overlap_percentage=0.5,
                                              page_decay_rate=0.1, batch_overlap_percentage=0.5):
    num_pages, num_samples = full_signal.shape
    batch_size = num_samples // num_batches
    effective_time_leap = batch_size - batch_overlap_percentage * batch_size

    index_map = []
    t_start_index = 0
    batch_index = 0

    while t_start_index < num_samples:
        t_end_index = min(t_start_index + batch_size, num_samples)
        index_map.append((batch_index, t_start_index, t_end_index))
        t_start_index += effective_time_leap
        batch_index += 1

    import time
    print(index_map)
    time.sleep(5)

    with ProcessPoolExecutor(max_workers=num_batches) as executer:
        futures = [
            executer.submit(
                process_block_2d_wavelet_decay_features_single_batch, full_signal, row_initial, n, batch_index, t_start_index, t_end_index, 
                num_pages, output_data_plot_dirs, wavelet, level, is_visualize, is_complex, 
                page_overlap_percentage, page_decay_rate
            )
            for batch_index, t_start_index, t_end_index in index_map
        ]

        # Wait for all futures to complete
        for future in futures:
            future.result()


##########################################
#          wavelet 2D - Features         #
##########################################

# Single block decay
def process_block_2d_wavelet_features_overlap(full_signal, row_initial, n, num_batches, output_data_plot_dirs, wavelet,
                                              level, is_visualize=False, is_complex=False, page_overlap_percentage=0.5,
                                              page_decay_rate=0.1, batch_overlap_percentage=0.5):
    """
    Processes blocks of the combined data using 2D Wavelet Transform, visualizes and saves the coefficients, and generates scalograms.

    Parameters:
    signal (np.ndarray): The 2D array where rows are pages and columns are time.
    n (int): Number of pages per block.
    t (int): Divisor for the time axis, to determine the width of the block in time.
    output_data_plot_dirs (list): Directories to save the output data and plots.
    wavelet (str): The wavelet to use for the 2D transform. Default is 'db4'.
    level (int): The level of decomposition for the 2D wavelet transform. Default is 2.
    is_complex (bool): If True, handles the data as complex; otherwise processes real and imaginary parts separately.
    is_visualize (bool): If True, visualizes the Scalogram and the Coeffs.
    """

    # Separate the output for data and plots
    output_data_dir, output_plot_dir = output_data_plot_dirs

    num_pages, num_samples = full_signal.shape
    batch_size = num_samples // num_batches

    start = 0 if row_initial is None else row_initial

    # Define the root directories based on wavelet type
    wavelet_root_data_dir = os.path.join(output_data_dir, f"{wavelet}_wavelet_{'complex' if is_complex else 'real'}")
    wavelet_root_plot_dir = os.path.join(output_plot_dir, f"{wavelet}_wavelet_{'complex' if is_complex else 'real'}")

    # Ensure the directories exist
    os.makedirs(wavelet_root_data_dir, exist_ok=True)
    os.makedirs(wavelet_root_plot_dir, exist_ok=True)

    overlap_pages = int(n * page_overlap_percentage)

    t_start_idx = 0
    batch_index = 0

    import time
    print("num_samples ", num_samples)
    time.sleep(5)

    # Function to wrap around the indices in the time or page axis
    def wrap_index(index, max_value):
        """ Returns a wrapped index to handle negative or out-of-bounds indices. """
        return index % max_value
    

    # effective_t_start = wrap_index(t_start_idx - overlap_batches, num_samples)
    effective_time_leap = batch_size - batch_overlap_percentage * batch_size

    finished_batches_flag = False

    # Circular overlap will work with negative indices - np.array.
    while (t_start_idx < num_samples) and (not finished_batches_flag):
        # TODO: Check negative modolu
        # Get effective block indices
        
        # effective_t_end   = wrap_index(effective_t_start + block_time_size, num_samples)
        # t_end_idx = wrap_index(t_start_idx + block_time_size, num_samples)

        # If it leakes from the samples, just move back the window by the leakage offset
        t_end_idx = min(t_start_idx + batch_size, num_samples)
        t_offset = 0 if ((t_start_idx + batch_size) <= num_samples) else ((t_start_idx + batch_size) - num_samples)
        t_start_idx -= t_offset
        
        # if t_offset != 0:
        #     finished_batches_flag = True
        
        # Define the current batch
        wavelet_batch_data_dir = os.path.join(wavelet_root_data_dir, f'batch_{batch_index}')
        wavelet_batch_plot_dir = os.path.join(wavelet_root_plot_dir, f'batch_{batch_index}')

        # Ensure the directories exist
        os.makedirs(wavelet_batch_data_dir, exist_ok=True)
        os.makedirs(wavelet_batch_plot_dir, exist_ok=True)

        # Ensure the loop termination condition
        if t_start_idx >= num_samples:
            finished_batches_flag = True

        for page_idx in range(start, num_pages, n):
            # TODO: Check negative modolu
            # Get effective block indices
            effective_p_start = int(wrap_index(int(page_idx - overlap_pages), num_pages))
            effective_p_end   = int(wrap_index(int(effective_p_start + n + overlap_pages), num_pages))

            if effective_p_end > effective_p_start:
                block = full_signal[effective_p_start:effective_p_end, t_start_idx:t_end_idx]
            else:
                print(effective_p_start, " ", num_pages, effective_p_start-num_pages)
                maxpage = int(num_pages)
                block_part1 = full_signal[effective_p_start:maxpage, t_start_idx:t_end_idx]
                
                # Second part from the start of the array to effective_p_end
                block_part2 = full_signal[0:effective_p_end, t_start_idx:t_end_idx]
                
                # Concatenate the two parts to form the full block
                block = np.vstack((block_part1, block_part2))
            
            # print("effective_p_start:effective_p_end, t_start_idx:t_end_idx")
            # print(effective_p_start, ":", effective_p_end, ", ", t_start_idx, ":", t_end_idx)
            # print("block size: ", block.shape, " N:", n)

            single_axis_decay(block, page_decay_rate, n)
            # print("decayed dims: ", block.shape)

            # Perform 2D wavelet transform on the block
            coeffs = process_wavelet_2d(block, wavelet=wavelet, level=level, is_complex=is_complex)

            # Create the directory for this batch of signals under the wavelet-specific directory
            batch_dir = os.path.join(wavelet_batch_data_dir, f'{wavelet}_2d_signal_{effective_p_start}_{effective_p_end}')
            os.makedirs(batch_dir, exist_ok=True)

            # Create directories for storing .npy and .txt files
            # npy_dir = os.path.join(batch_dir, 'npy')
            # text_dir = os.path.join(batch_dir, 'text')
            features_dir = os.path.join(batch_dir, 'features')
            # os.makedirs(npy_dir, exist_ok=True)
            # os.makedirs(text_dir, exist_ok=True)
            os.makedirs(features_dir)

            # # Create directories for storing real and image files
            # npy_dir_real = os.path.join(npy_dir, 'mag')
            # npy_dir_imag = os.path.join(npy_dir, 'phase')
            # os.makedirs(npy_dir_real, exist_ok=True)
            # os.makedirs(npy_dir_imag, exist_ok=True)

            # txt_dir_real = os.path.join(text_dir, 'mag')
            # txt_dir_imag = os.path.join(text_dir, 'phase')
            # os.makedirs(txt_dir_real, exist_ok=True)
            # os.makedirs(txt_dir_imag, exist_ok=True)

            # Visualize
            if is_visualize:
                # Create a subdirectory in the plot output directory for visualizations
                signal_plot_dir = os.path.join(wavelet_batch_plot_dir, f'visualize_1d_signal_{effective_p_start}_{effective_p_end}')
                os.makedirs(signal_plot_dir, exist_ok=True)

                # Call the existing visualize functions to save the plots
                visualize_wavelet_1d(coeffs, effective_p_start, signal_plot_dir)
                create_scalogram_1d(wavelet, block, effective_p_start, signal_plot_dir)

            # Exctract all possible features
            features = block_feature_extractor.process_wavelet_coeffs_block(coeffs, feature_type_list, num_levels=level, coeffs2=None, coeffs_next_batch=None)

            create_files(features, features_dir)

            print(f"Processed wavelet for block [{t_start_idx}, {t_end_idx}] X [{effective_p_start}, {effective_p_end}]")
    
        # Zip the batch directory after processing is complete
        zip_output_path = os.path.join(wavelet_root_data_dir, f'batch_{batch_index}.zip')
        zip_dir(wavelet_batch_data_dir, zip_output_path)

        # Remove the original directory after zipping
        shutil.rmtree(wavelet_batch_data_dir)
        print(f"Removed original directory: {wavelet_batch_data_dir}")
        t_start_idx += int(effective_time_leap)
        batch_index+=1


def exponential_decay(size, decay_rate=0.1):
    """
    Generates an exponential decay matrix.
    
    Parameters:
    size (tuple): Size of the block (pages, time points).
    decay_rate (float): The rate of decay for the exponential function.
    
    Returns:
    np.ndarray: The decay matrix.
    """
    pages, time_points = size
    decay = np.exp(-decay_rate * np.arange(pages))[:, None] * np.exp(-decay_rate * np.arange(time_points))
    return decay


def single_axis_decay(block, page_decay_rate, real_page_size):
    # Get effective size of block
    effective_p_size, effective_t_size = block.shape

    # Get the head and tail size to be decayed, ensure real_page_size < effective_p_size
    if real_page_size >= effective_p_size:
        raise ValueError("real_page_size must be smaller than effective_p_size")

    # Get the head and tail size to be decayed
    head_tail_size = (effective_p_size - real_page_size) // 2

    for jmp in range(0, head_tail_size):
        head_idx = head_tail_size - jmp - 1
        tail_idx = effective_p_size - (head_tail_size - jmp)

        for curr_sample in range(0, effective_t_size):
            decay_factor = np.exp(-page_decay_rate * (jmp+1))
            block[head_idx, curr_sample] = block[head_idx, curr_sample] * decay_factor
            block[tail_idx, curr_sample] = block[tail_idx, curr_sample] * decay_factor


def create_scalogram_3d_plotly(block, block_index, block_index_p, block_index_t, output_data_plot_dirs, is_complex=False):
    """
    Creates and saves a 3D scalogram from the 2D block using Plotly.

    Parameters:
    block (np.ndarray): The 2D data array (block).
    block_index (int): The index of the current block being processed.
    output_data_plot_dirs (list): Directories to save the output data and plots.
    is_complex (bool): If True, handles the data as complex; otherwise processes real and imaginary parts separately.
    """
    print(f"Creating 3D scalogram for block {block_index}")

    x = np.arange(block.shape[1])  # Time axis
    y = np.arange(block.shape[0])  # Pages axis
    X, Y = np.meshgrid(x, y)
    Z = np.abs(block) if is_complex else block

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    fig.update_layout(
        title=f'3D Scalogram - Block {block_index}',
        scene=dict(
            xaxis_title='Time',
            yaxis_title='Pages',
            zaxis_title='Magnitude'
        )
    )

    fig.write_html(os.path.join(output_data_plot_dirs[1], f'scalogram_block_{block_index}__{block_index_p}_{block_index_t}.html'))

    print(f'3D Scalogram for Block {block_index} processed and images saved.')


def visualize_wavelet_3d_plotly(coeffs, block_index, block_index_p, block_index_t, output_data_plot_dirs, visualize, is_complex=False):
    """
    Visualizes and saves the wavelet coefficients for a 2D block as a 3D surface plot using Plotly.

    Parameters:
    coeffs (list or tuple): The wavelet coefficients from the 2D wavelet transform.
    block_index (int): The index of the current block being processed.
    output_data_plot_dirs (list): Directories to save the output data and plots.
    visualize (bool): Whether to visualize the coefficients.
    is_complex (bool): If True, the data is treated as complex; otherwise, the real and imaginary parts are processed separately.
    """
    if is_complex:
        real_coeffs, imag_coeffs = coeffs
        # print(real_coeffs)  # For debugging
        # print("------")
        # print(imag_coeffs)  # For debugging

        if visualize:
            for i, (real_coeff, imag_coeff) in enumerate(zip(real_coeffs, imag_coeffs)):
                if i == 0:  # Approximation coefficients
                    x = np.arange(real_coeff.shape[1])
                    y = np.arange(real_coeff.shape[0])
                    X, Y = np.meshgrid(x, y)

                    fig = go.Figure(data=[go.Surface(z=np.abs(real_coeff), x=X, y=Y, colorscale='Viridis')])
                    fig.update_layout(
                        title=f'3D Wavelet Real Approximation Coefficients - Block {block_index_p}_{block_index_t} Level {i}',
                        scene=dict(
                            xaxis_title='Time',
                            yaxis_title='Pages',
                            zaxis_title='Magnitude'
                        )
                    )
                    fig.write_html(os.path.join(output_data_plot_dirs[1], f'wavelet_real_approx_3d_block_{block_index_p}_{block_index_t}_level_{i}.html'))

                    fig = go.Figure(data=[go.Surface(z=np.abs(imag_coeff), x=X, y=Y, colorscale='Viridis')])
                    fig.update_layout(
                        title=f'3D Wavelet Imaginary Approximation Coefficients - Block {block_index_p}_{block_index_t} Level {i}',
                        scene=dict(
                            xaxis_title='Time',
                            yaxis_title='Pages',
                            zaxis_title='Magnitude'
                        )
                    )
                    fig.write_html(os.path.join(output_data_plot_dirs[1], f'wavelet_imag_approx_3d_block_{block_index_p}_{block_index_t}_level_{i}.html'))
                else:  # Detail coefficients (tuple of (cH, cV, cD))
                    cH_real, cV_real, cD_real = real_coeff
                    cH_imag, cV_imag, cD_imag = imag_coeff

                    for coeff_name, real_c, imag_c in zip(['Horizontal', 'Vertical', 'Diagonal'], [cH_real, cV_real, cD_real], [cH_imag, cV_imag, cD_imag]):
                        x = np.arange(real_c.shape[1])
                        y = np.arange(real_c.shape[0])
                        X, Y = np.meshgrid(x, y)

                        fig = go.Figure(data=[go.Surface(z=np.abs(real_c), x=X, y=Y, colorscale='Viridis')])
                        fig.update_layout(
                            title=f'3D Wavelet Real {coeff_name} Coefficients - Block {block_index_p}_{block_index_t} Level {i}',
                            scene=dict(
                                xaxis_title='Time',
                                yaxis_title='Pages',
                                zaxis_title='Magnitude'
                            )
                        )
                        fig.write_html(os.path.join(output_data_plot_dirs[1], f'wavelet_real_{coeff_name.lower()}_3d_block_{block_index_p}_{block_index_t}_level_{i}.html'))

                        fig = go.Figure(data=[go.Surface(z=np.abs(imag_c), x=X, y=Y, colorscale='Viridis')])
                        fig.update_layout(
                            title=f'3D Wavelet Imaginary {coeff_name} Coefficients - Block {block_index_p}_{block_index_t} Level {i}',
                            scene=dict(
                                xaxis_title='Time',
                                yaxis_title='Pages',
                                zaxis_title='Magnitude'
                            )
                        )
                        fig.write_html(os.path.join(output_data_plot_dirs[1], f'wavelet_imag_{coeff_name.lower()}_3d_block_{block_index_p}_{block_index_t}_level_{i}.html'))

        # Save the wavelet coefficients as data
        for i, (real_coeff, imag_coeff) in enumerate(zip(real_coeffs, imag_coeffs)):
            if i == 0:  # Approximation coefficients
                save_analysis_data(np.abs(real_coeff), os.path.join(output_data_plot_dirs[0], f'wavelet_real_approx_coeffs_block_{block_index_p}_{block_index_t}_level_{i}.npy'))
                save_analysis_data(np.abs(imag_coeff), os.path.join(output_data_plot_dirs[0], f'wavelet_imag_approx_coeffs_block_{block_index_p}_{block_index_t}_level_{i}.npy'))
            else:  # Detail coefficients
                cH_real, cV_real, cD_real = real_coeff
                cH_imag, cV_imag, cD_imag = imag_coeff
                for coeff_name, real_c, imag_c in zip(['Horizontal', 'Vertical', 'Diagonal'], [cH_real, cV_real, cD_real], [cH_imag, cV_imag, cD_imag]):
                    save_analysis_data(np.abs(real_c), os.path.join(output_data_plot_dirs[0], f'wavelet_real_{coeff_name.lower()}_coeffs_block_{block_index_p}_{block_index_t}_level_{i}.npy'))
                    save_analysis_data(np.abs(imag_c), os.path.join(output_data_plot_dirs[0], f'wavelet_imag_{coeff_name.lower()}_coeffs_block_{block_index_p}_{block_index_t}_level_{i}.npy'))

    else:
        if visualize:
            cA = coeffs[0]  # Approximation coefficients
            cH, cV, cD = coeffs[1]  # Horizontal, Vertical, Diagonal detail coefficients

            x = np.arange(cA.shape[1])
            y = np.arange(cA.shape[0])
            X, Y = np.meshgrid(x, y)

            fig = go.Figure(data=[go.Surface(z=cA, x=X, y=Y, colorscale='Viridis')])
            fig.update_layout(
                title=f'3D Wavelet Approximation Coefficients - Block {block_index_p}_{block_index_t}',
                scene=dict(
                    xaxis_title='Time',
                    yaxis_title='Pages',
                    zaxis_title='Magnitude'
                )
            )
            fig.write_html(os.path.join(output_data_plot_dirs[1], f'wavelet_approx_3d_block_{block_index_p}_{block_index_t}.html'))

            # Repeat for detail coefficients cH, cV, cD
            detail_coeffs = {'Horizontal': cH, 'Vertical': cV, 'Diagonal': cD}
            for coeff_name, coeff in detail_coeffs.items():
                fig = go.Figure(data=[go.Surface(z=coeff, x=X, y=Y, colorscale='Viridis')])
                fig.update_layout(
                    title=f'3D Wavelet {coeff_name} Detail Coefficients - Block {block_index_p}_{block_index_t}',
                    scene=dict(
                        xaxis_title='Time',
                        yaxis_title='Pages',
                        zaxis_title='Magnitude'
                    )
                )
                fig.write_html(os.path.join(output_data_plot_dirs[1], f'wavelet_{coeff_name.lower()}_3d_block_{block_index_p}_{block_index_t}.html'))

        # Save the wavelet coefficients as data
        for i, coeff in enumerate(coeffs):
            save_analysis_data(coeff, os.path.join(output_data_plot_dirs[0], f'wavelet_coeffs_block_{block_index_p}_{block_index_t}_level_{i}.npy'))


def save_analysis_data(data, filename):
    """Helper function to save numpy array to file."""
    np.save(filename, data)


def process_2d_scattering_wavelets_with_scalogram(__npy_only, row_initial, combined_data, n, t, output_data_plot_dirs, is_complex, is_visualize, wavelet, J=4, L=8):
    """
    Processes blocks of the combined data using 2D Wavelet Scattering, visualizes and saves the coefficients, and generates scalograms.

    Parameters:
    combined_data (np.ndarray): The 2D array where rows are pages and columns are time.
    n (int): Number of pages per block.
    t (int): Divisor for the time axis, to determine the width of the block in time.
    output_data_plot_dirs (list): Directories to save the output data and plots.
    J (int): Number of scales for the 2D scattering. Default is 4.
    L (int): Number of angles for the 2D scattering. Default is 8.
    """ 
    print("process_2d_scattering_wavelets_with_scalogram")
    # Separate the output for data and plots
    output_data_dir, output_plot_dir = output_data_plot_dirs

    num_pages, num_time_points = combined_data.shape
    block_time_size = num_time_points // t

    start = 0 if row_initial is None else row_initial

    # Define the root directories based on wavelet type
    wavelet_root_data_dir = os.path.join(output_data_dir, f"{wavelet}_wavelet_{'complex' if is_complex else 'real'}")
    wavelet_root_plot_dir = os.path.join(output_plot_dir, f"{wavelet}_wavelet_{'complex' if is_complex else 'real'}")

    # Ensure the directories exist
    os.makedirs(wavelet_root_data_dir, exist_ok=True)
    os.makedirs(wavelet_root_plot_dir, exist_ok=True)

    for batch in range (0, t):
        # Define the current batch
        wavelet_batch_data_dir = os.path.join(wavelet_root_data_dir, f'batch_{batch}')
        wavelet_batch_plot_dir = os.path.join(wavelet_root_plot_dir, f'batch_{batch}')

        if os.path.exists(os.path.join(wavelet_batch_data_dir, f'batch_{batch}.zip')):
            print(os.path.join(wavelet_batch_data_dir, f'batch_{batch}.zip'), " exists!")
            continue

        # Ensure the directories exist
        os.makedirs(wavelet_batch_data_dir, exist_ok=True)
        os.makedirs(wavelet_batch_plot_dir, exist_ok=True)

        # Define the edges of samples of the current batch
        t_start = block_time_size * batch
        t_end = t_start + block_time_size - 1

        for signal_index in range(start, num_pages, n):
            # Define the edges of samples of the current batch
            p_start = signal_index
            p_end = p_start + n - 1
            print('startt ', t_start, ' endt ', t_end )
            print('startp ', p_start, ' endp ', p_end )
            # print('numpages ', num_pages, ' signal_index ', signal_index, ' n ', n )
            # Ensure we don't go out of bounds
            signal = combined_data[p_start:(p_end+1), t_start:(t_end+1)]

            # Separate real and imaginary parts
            real_signal = np.real(signal)
            imag_signal = np.imag(signal)

            # Convert both parts to tensors
            real_tensor = torch.from_numpy(real_signal).float().unsqueeze(0).unsqueeze(0)
            imag_tensor = torch.from_numpy(imag_signal).float().unsqueeze(0).unsqueeze(0)

            # Perform 2D wavelet scattering on the block
            scattering = Scattering2D(J=J, shape=signal.shape)  # Initialize the scattering object with the block shape
            
            # Process both parts through scattering
            coeffs_real = scattering(real_tensor)
            coeffs_imag = scattering(imag_tensor)

            batch_dir = os.path.join(wavelet_batch_data_dir, f'{wavelet}_2d_signal_{p_start}_{p_end}')
            file_dir = os.path.join(wavelet_batch_data_dir, f'{p_start}_{p_end}.zip')

            if os.path.exists(file_dir):
                print(file_dir," exists")
                continue
            os.makedirs(batch_dir, exist_ok=True)

            scatter_create_files([coeffs_real, coeffs_imag], batch_dir)    

            # Zip the batch directory after processing is complete
            zip_output_path = os.path.join(wavelet_batch_data_dir, f'{wavelet}_2d_signal_{p_start}_{p_end}.zip')
            zip_dir(batch_dir, zip_output_path)

            subprocess.run(['rm', '-f', batch_dir], check=True)
            print(f"Removed original directory: {batch_dir}")

            print(f"Processed wavelet for block [{t_start}, {t_start}] X [{p_start}, {p_end}]")

        # Zip the batch directory after processing is complete
        zip_output_path = os.path.join(wavelet_root_data_dir, f'batch_{batch}.zip')
        zip_dir(wavelet_batch_data_dir, zip_output_path)

        # Remove the original directory after zipping
        # shutil.rmtree(wavelet_batch_data_dir)
        subprocess.run(['rm', '-rf', wavelet_batch_data_dir], check=True)
        print(f"Removed original directory: {wavelet_batch_data_dir}")

        

                

            # # print(coeffs_real.shape)
            # # print(coeffs_real)
            # # print('---------------------------------------------')
            # # print(coeffs_imag.shape)
            # # print(coeffs_imag)

            # # Combine the real and imaginary coefficients if needed
            # coeffs = (coeffs_real, coeffs_imag)

            # # Create the directory for this batch of signals under the wavelet-specific directory
            # batch_dir = os.path.join(wavelet_batch_data_dir, f'{wavelet}_2d_signal_{p_start}_{p_end}')
            # os.makedirs(batch_dir, exist_ok=True)

            # # Create directories for storing .npy and .txt files
            # npy_dir = os.path.join(batch_dir, 'npy')
            # text_dir = os.path.join(batch_dir, 'text')
            # os.makedirs(npy_dir, exist_ok=True)
            # os.makedirs(text_dir, exist_ok=True)

            # # Create directories for storing real and image files
            # npy_dir_real = os.path.join(npy_dir, 'real')
            # npy_dir_imag = os.path.join(npy_dir, 'image')
            # os.makedirs(npy_dir_real, exist_ok=True)
            # os.makedirs(npy_dir_imag, exist_ok=True)

            # txt_dir_real = os.path.join(text_dir, 'real')
            # txt_dir_imag = os.path.join(text_dir, 'image')
            # os.makedirs(txt_dir_real, exist_ok=True)
            # os.makedirs(txt_dir_imag, exist_ok=True)

            # # Loop over each scattering path of real data
            # for path_idx in range(coeffs_real.shape[2]):  # Loop over scattering paths
            #     # Extract coefficients for the current path
            #     path_coeffs_real = coeffs_real[0, 0, path_idx, :, :]  # Shape: (height, width), assuming batch size is 1

            #     print("saving npy real")
            #     # Save real part as .npy
            #     np.save(os.path.join(npy_dir_real, f'scattering_coeffs_path_{path_idx}_real.npy'), path_coeffs_real.cpu().numpy())
                
            #     if not __npy_only:
            #         print("saving text real")
            #         # Save real part in human-readable .txt format
            #         with open(os.path.join(txt_dir_real, f'scattering_coeffs_path_{path_idx}_real.txt'), 'w') as f:
            #             f.write(f'===== Scattering Coefficients (Real Part) for Path {path_idx} =====\n')
            #             f.write(f'Coefficient dimensions: {path_coeffs_real.shape}\n')
            #             f.write(f'Path Index: {path_idx}\n')
            #             f.write('Coefficients:\n')
            #             np.savetxt(f, path_coeffs_real.cpu().numpy(), fmt='%f')
            
            # if coeffs_imag is not None:
            #     # Loop over each scattering path of imaginary data
            #     for path_idx in range(coeffs_imag.shape[1]):  # Loop over scattering paths
            #         # Extract coefficients for the current path
            #         path_coeffs_image = coeffs_imag[0, 0, path_idx, :, :]  # Shape: (height, width), assuming batch size is 1

            #         print("saving npy real")
            #         np.save(os.path.join(npy_dir_imag, f'scattering_coeffs_path_{path_idx}_imag.npy'), path_coeffs_image.cpu().numpy())
                    
            #         if not __npy_only:
            #             # Save imaginary part in human-readable .txt format
            #             print("saving text real")
            #             with open(os.path.join(txt_dir_imag, f'scattering_coeffs_path_{path_idx}_imag.txt'), 'w') as f:
            #                 f.write(f'===== Scattering Coefficients (Imaginary Part) for Path {path_idx} =====\n')
            #                 f.write(f'Coefficient dimensions: {path_coeffs_image.shape}\n')
            #                 f.write(f'Path Index: {path_idx}\n')
            #                 f.write('Coefficients:\n')
            #                 np.savetxt(f, path_coeffs_image.cpu().numpy(), fmt='%f')


            # if (is_visualize) and (signal_index % 1024 == 0):
            #     # Create a subdirectory in the plot output directory for visualizations
            #     signal_plot_dir = os.path.join(wavelet_batch_plot_dir, f'visualize_1d_signal_{p_start}_{p_end}')
            #     os.makedirs(signal_plot_dir, exist_ok=True)

            #     # Call the existing visualize functions to save the plots
            #     visualize_wavelet_1d(coeffs, signal_index, signal_plot_dir)
            #     create_scalogram_1d(wavelet, signal, signal_index, signal_plot_dir)
            
            # print(f"Processed wavelet for block [{t_start}, {t_start}] X [{p_start}, {p_end}]")


def process_2d_scattering_parallel_overlap_single(full_signal, row_initial, n, batch_index, t_start_idx_flt, t_end_idx__flt, num_pages, output_data_plot_dirs, wavelet, 
                  page_overlap_percentage, page_decay_rate, J):
    # Separate the output for data and plots
    output_data_dir, output_plot_dir = output_data_plot_dirs
   
    t_start_idx = int(t_start_idx_flt)
    t_end_idx = int(t_end_idx__flt)
    overlap_pages = int(n * page_overlap_percentage)
    start = 0 if row_initial is None else row_initial

    # Define the root directories based on wavelet type
    scatter_root_data_dir = os.path.join(output_data_dir, f'scatter')
    scatter_root_plot_dir = os.path.join(output_plot_dir, f'scatter')

    # Ensure the directories exist
    os.makedirs(scatter_root_data_dir, exist_ok=True)
    os.makedirs(scatter_root_plot_dir, exist_ok=True)

    # Define the current batch
    wavelet_batch_data_dir = os.path.join(scatter_root_data_dir, f'batch_{batch_index}')
    wavelet_batch_plot_dir = os.path.join(scatter_root_plot_dir, f'batch_{batch_index}')

    if os.path.exists(os.path.join(scatter_root_data_dir, f'batch_{batch_index}.zip')):
        print(os.path.join(scatter_root_data_dir, f'batch_{batch_index}.zip'), " exists!")
        return

    # Ensure the directories exist
    os.makedirs(wavelet_batch_data_dir, exist_ok=True)
    os.makedirs(wavelet_batch_plot_dir, exist_ok=True)

    # Function to wrap around the indices in the time or page axis
    def wrap_index(index, max_value):
        """ Returns a wrapped index to handle negative or out-of-bounds indices. """
        return index % max_value

    for page_idx in range(start, num_pages, n):
        effective_p_start = int(wrap_index(int(page_idx - overlap_pages), num_pages))
        effective_p_end   = int(wrap_index(int(effective_p_start + n + overlap_pages), num_pages))

        if effective_p_end > effective_p_start:
            block = full_signal[effective_p_start:effective_p_end, t_start_idx:t_end_idx]
        else:
            print(effective_p_start, " ", num_pages, effective_p_start-num_pages)

            # First part from effective_p_start to the end of the array
            maxpage = int(num_pages) 
            block_part1 = full_signal[effective_p_start:maxpage, t_start_idx:t_end_idx]
            
            # Second part from the start of the array to effective_p_end
            block_part2 = full_signal[0:effective_p_end, t_start_idx:t_end_idx]
            
            # Concatenate the two parts to form the full block
            block = np.vstack((block_part1, block_part2))
        
        # batch_dir = os.path.join(wavelet_batch_data_dir, f'{effective_p_start}_{effective_p_end}')
        file_dir = os.path.join(wavelet_batch_data_dir, f'{effective_p_start}_{effective_p_end}_{t_start_idx}_{t_end_idx}_features.h5')

        if os.path.exists(file_dir):
            print(file_dir," exists")
            continue

        # os.makedirs(batch_dir, exist_ok=True)


        single_axis_decay(block, page_decay_rate, n)

    # Separate real and imaginary parts
        real_signal = np.real(block)
        imag_signal = np.imag(block)

        # Convert both parts to tensors
        real_tensor = torch.from_numpy(real_signal).float().unsqueeze(0).unsqueeze(0)
        imag_tensor = torch.from_numpy(imag_signal).float().unsqueeze(0).unsqueeze(0)

        # Perform 2D wavelet scattering on the block
        scattering = Scattering2D(J=J, shape=block.shape)  # Initialize the scattering object with the block shape
        
        # Process both parts through scattering
        coeffs_real = scattering(real_tensor)
        coeffs_imag = scattering(imag_tensor)

        # Process both parts through scattering
        coeffs_real = scattering(real_tensor)
        coeffs_imag = scattering(imag_tensor)
        

        scatter_create_files([coeffs_real, coeffs_imag], wavelet_batch_data_dir, effective_p_start, effective_p_end, t_start_idx, t_end_idx)

        # Zip the batch directory after processing is complete
        # zip_output_path = os.path.join(wavelet_batch_data_dir, f'{effective_p_start}_{effective_p_end}.zip')
        # zip_dir(batch_dir, zip_output_path)

        # subprocess.run(['rm', '-rf', batch_dir], check=True)
        # print(f"Removed original directory: {batch_dir}")

        print(f"Processed wavelet for block [{t_start_idx}, {t_end_idx}] X [{effective_p_start}, {effective_p_end}]")

    # Zip the batch directory after processing is complete
    zip_output_path = os.path.join(scatter_root_data_dir, f'batch_{batch_index}.zip')
    zip_dir(wavelet_batch_data_dir, zip_output_path)

    # Remove the original directory after zipping
    # shutil.rmtree(wavelet_batch_data_dir)
    subprocess.run(['rm', '-rf', wavelet_batch_data_dir], check=True)
    print(f"Removed original directory: {wavelet_batch_data_dir}")


def process_2d_scattering_parallel_overlap(__npy_only, is_visualize, row_initial, combined_data, N, num_batches,
                                                           output_data_plot_dirs, is_complex, wavelet, J=4, L=8, 
                                                           page_overlap_percentage=0.5, page_decay_rate=0.1, 
                                                           batch_overlap_percentage=0.5):
    print("parallel - scatter - exp")
    num_pages, num_samples = combined_data.shape
    block_time_size = num_samples // num_batches

    effective_time_leap = block_time_size - batch_overlap_percentage * block_time_size

    index_map = []
    t_start_index = 0
    batch_index = 0

    while t_start_index < num_samples:
        t_end_index = min(t_start_index + block_time_size, num_samples-1)
        index_map.append((batch_index, t_start_index, t_end_index))
        t_start_index += effective_time_leap
        batch_index += 1

    with ProcessPoolExecutor(max_workers=num_batches) as executer:
        futures = [
            executer.submit(
                process_2d_scattering_parallel_overlap_single, combined_data, row_initial, N, batch_index, t_start_index, t_end_index, num_pages, output_data_plot_dirs, wavelet, 
                page_overlap_percentage, page_decay_rate, J=J
            )
            for batch_index, t_start_index, t_end_index in index_map 
        ]

        #  Wait for all futures to complete
        for future in futures:
            future.result()


def process_2d_scattering_wavelets_with_scalogram_overlap(__npy_only, is_visualize, row_initial, combined_data, N, num_batches,
                                                           output_data_plot_dirs, is_complex, wavelet, J=4, L=8, 
                                                           page_overlap_percentage=0.5, page_decay_rate=0.1, 
                                                           batch_overlap_percentage=0.5):
    """
    Processes blocks of the combined data using 2D Wavelet Scattering, visualizes and saves the coefficients, and generates scalograms.

    Parameters:
    combined_data (np.ndarray): The 2D array where rows are pages and columns are time.
    n (int): Number of pages per block.
    t (int): Divisor for the time axis, to determine the width of the block in time.
    output_data_plot_dirs (list): Directories to save the output data and plots.
    J (int): Number of scales for the 2D scattering. Default is 4.
    L (int): Number of angles for the 2D scattering. Default is 8.
    """ 
    print("process_2d_scattering_wavelets_with_scalogram")
    # Separate the output for data and plots
    output_data_dir, output_plot_dir = output_data_plot_dirs

    num_pages, num_samples = combined_data.shape
    block_time_size = num_samples // num_batches

    start = 0 if row_initial is None else row_initial

    # Define the root directories based on wavelet type
    scatter_wavelet_root_data_dir = os.path.join(output_data_dir, f'scatter')
    scatter_wavelet_root_plot_dir = os.path.join(output_plot_dir, f'scatter')

    # Ensure the directories exist
    os.makedirs(scatter_wavelet_root_data_dir, exist_ok=True)
    os.makedirs(scatter_wavelet_root_plot_dir, exist_ok=True)

    overlap_pages = int(N * page_overlap_percentage)
    overlap_batches = int(block_time_size * batch_overlap_percentage)

    t_start_idx = 0
    batch_index = 0

    # Function to wrap around the indices in the time or page axis
    def wrap_index(index, max_value):
        """ Returns a wrapped index to handle negative or out-of-bounds indices. """
        return index % max_value
    
    # effective_t_start = wrap_index(t_start_idx - overlap_batches, num_samples)
    effective_time_leap = block_time_size - batch_overlap_percentage * block_time_size

    finished_batches_flag = False

    # Circular overlap will work with negative indices - np.array.
    while (t_start_idx < num_samples) and (not finished_batches_flag):
        # If it leakes from the samples, just move back the window by the leakage offset
        t_end_idx = min(t_start_idx + block_time_size, num_samples)
        t_offset = 0 if ((t_start_idx + block_time_size) <= num_samples) else ((t_start_idx + block_time_size) - num_samples)
        t_start_idx -= t_offset
        
        if t_offset != 0:
            finished_batches_flag = True

        # Define the current batch
        wavelet_batch_data_dir = os.path.join(scatter_wavelet_root_data_dir, f'batch_{batch_index}')
        wavelet_batch_plot_dir = os.path.join(scatter_wavelet_root_plot_dir, f'batch_{batch_index}')

        if os.path.exists(os.path.join(wavelet_batch_data_dir, f'batch_{batch_index}.zip')):
            print(os.path.join(wavelet_batch_data_dir, f'batch_{batch_index}.zip'), " exists!")
            t_start_idx += int(effective_time_leap)
            batch_index += 1
            continue

        # Ensure the directories exist
        os.makedirs(wavelet_batch_data_dir, exist_ok=True)
        os.makedirs(wavelet_batch_plot_dir, exist_ok=True)

        # Define the edges of samples of the current batch
        t_start = block_time_size * batch_index
        t_end = t_start + block_time_size - 1

        for page_idx in range(start, num_pages, N):

            effective_p_start = int(wrap_index(int(page_idx - overlap_pages), num_pages))
            effective_p_end   = int(wrap_index(int(effective_p_start + N + overlap_pages), num_pages))

            if effective_p_end > effective_p_start:
                block = combined_data[effective_p_start:effective_p_end, t_start_idx:t_end_idx]
            else:
                print(effective_p_start, " ", num_pages, effective_p_start-num_pages)

                # First part from effective_p_start to the end of the array
                maxpage = int(num_pages) 
                block_part1 = combined_data[effective_p_start:maxpage, t_start_idx:t_end_idx]
                
                # Second part from the start of the array to effective_p_end
                block_part2 = combined_data[0:effective_p_end, t_start_idx:t_end_idx]
                
                # Concatenate the two parts to form the full block
                block = np.vstack((block_part1, block_part2))

            print("effective_p_start:effective_p_end, t_start_idx:t_end_idx")
            print(effective_p_start, ":", effective_p_end, ", ", t_start_idx, ":", t_end_idx)
            print("block size: ", block.shape, " N:", N)

            single_axis_decay(block, page_decay_rate, N)
            print("decayed dims: ", block.shape)

            # Separate real and imaginary parts
            real_signal = np.real(block)
            imag_signal = np.imag(block)

            # Convert both parts to tensors
            real_tensor = torch.from_numpy(real_signal).float().unsqueeze(0).unsqueeze(0)
            imag_tensor = torch.from_numpy(imag_signal).float().unsqueeze(0).unsqueeze(0)

            # Perform 2D wavelet scattering on the block
            scattering = Scattering2D(J=J, shape=block.shape)  # Initialize the scattering object with the block shape
            
            # Process both parts through scattering
            coeffs_real = scattering(real_tensor)
            coeffs_imag = scattering(imag_tensor)

            # Process both parts through scattering
            coeffs_real = scattering(real_tensor)
            coeffs_imag = scattering(imag_tensor)

            # batch_dir = os.path.join(wavelet_batch_data_dir, f'{effective_p_start}_{effective_p_end}')
            file_dir = os.path.join(wavelet_batch_data_dir, f'{effective_p_start}_{effective_p_end}_{t_start}_{t_end}_features.h5')

            if os.path.exists(file_dir):
                print(file_dir," exists")
                continue

            # os.makedirs(batch_dir, exist_ok=True)

            scatter_create_files([coeffs_real, coeffs_imag], wavelet_batch_data_dir, effective_p_start ,effective_p_end, t_start, t_end)

            # Zip the batch directory after processing is complete
            # zip_output_path = os.path.join(wavelet_batch_data_dir, f'{effective_p_start}_{effective_p_end}.zip')
            # zip_dir(wavelet_batch_data_dir, file_dir)

            # # Zip the batch directory after processing is complete
            # zip_output_path = os.path.join(wavelet_batch_data_dir, f'{effective_p_start}_{effective_p_end}.zip')
            # zip_dir(batch_dir, zip_output_path)

            # subprocess.run(['rm', '-rf', batch_dir], check=True)
            # print(f"Removed original directory: {batch_dir}")

            print(f"Processed wavelet for block [{t_start}, {t_end}] X [{effective_p_start}, {effective_p_end}]")

        # Zip the batch directory after processing is complete
        zip_output_path = os.path.join(scatter_wavelet_root_data_dir, f'batch_{batch_index}.zip')
        zip_dir(wavelet_batch_data_dir, zip_output_path)

        # Remove the original directory after zipping
        # shutil.rmtree(wavelet_batch_data_dir)
        subprocess.run(['rm', '-rf', wavelet_batch_data_dir], check=True)
        print(f"Removed original directory: {wavelet_batch_data_dir}")

        t_start_idx += int(effective_time_leap)
        batch_index += 1
        
            # # print(coeffs_real.shape)
            # # print(coeffs_real)
            # # print('---------------------------------------------')
            # # print(coeffs_imag.shape)
            # # print(coeffs_imag)

            # # Combine the real and imaginary coefficients if needed
            # coeffs = (coeffs_real, coeffs_imag)

            # # Create the directory for this batch of signals under the wavelet-specific directory
            # batch_dir = os.path.join(wavelet_batch_data_dir, f'{wavelet}_2d_signal_[{effective_p_start},{effective_p_end}]_[{page_idx},{page_idx+N}]')
            # os.makedirs(batch_dir, exist_ok=True)

            # # Create directories for storing .npy and .txt files
            # npy_dir = os.path.join(batch_dir, 'npy')
            # text_dir = os.path.join(batch_dir, 'text')
            # os.makedirs(npy_dir, exist_ok=True)
            # os.makedirs(text_dir, exist_ok=True)

            # # Create directories for storing real and image files
            # npy_dir_real = os.path.join(npy_dir, 'real')
            # npy_dir_imag = os.path.join(npy_dir, 'image')
            # os.makedirs(npy_dir_real, exist_ok=True)
            # os.makedirs(npy_dir_imag, exist_ok=True)

            # txt_dir_real = os.path.join(text_dir, 'real')
            # txt_dir_imag = os.path.join(text_dir, 'image')
            # os.makedirs(txt_dir_real, exist_ok=True)
            # os.makedirs(txt_dir_imag, exist_ok=True)

            # # Loop over each scattering path of real data
            # for path_idx in range(coeffs_real.shape[2]):  # Loop over scattering paths
            #     # Extract coefficients for the current path
            #     path_coeffs_real = coeffs_real[0, 0, path_idx, :, :]  # Shape: (height, width), assuming batch size is 1

            #     print("saving npy real")
            #     # Save real part as .npy
            #     np.save(os.path.join(npy_dir_real, f'scattering_coeffs_path_{path_idx}_real.npy'), path_coeffs_real.cpu().numpy())
                
            #     if not __npy_only:
            #         print("saving text real")
            #         # Save real part in human-readable .txt format
            #         with open(os.path.join(txt_dir_real, f'scattering_coeffs_path_{path_idx}_real.txt'), 'w') as f:
            #             f.write(f'===== Scattering Coefficients (Real Part) for Path {path_idx} =====\n')
            #             f.write(f'Coefficient dimensions: {path_coeffs_real.shape}\n')
            #             f.write(f'Path Index: {path_idx}\n')
            #             f.write('Coefficients:\n')
            #             np.savetxt(f, path_coeffs_real.cpu().numpy(), fmt='%f')
            
            # if coeffs_imag is not None:
            #     # Loop over each scattering path of imaginary data
            #     for path_idx in range(coeffs_imag.shape[1]):  # Loop over scattering paths
            #         # Extract coefficients for the current path
            #         path_coeffs_image = coeffs_imag[0, 0, path_idx, :, :]  # Shape: (height, width), assuming batch size is 1

            #         print("saving npy real")
            #         np.save(os.path.join(npy_dir_imag, f'scattering_coeffs_path_{path_idx}_imag.npy'), path_coeffs_image.cpu().numpy())
                    
            #         if not __npy_only:
            #             # Save imaginary part in human-readable .txt format
            #             print("saving text real")
            #             with open(os.path.join(txt_dir_imag, f'scattering_coeffs_path_{path_idx}_imag.txt'), 'w') as f:
            #                 f.write(f'===== Scattering Coefficients (Imaginary Part) for Path {path_idx} =====\n')
            #                 f.write(f'Coefficient dimensions: {path_coeffs_image.shape}\n')
            #                 f.write(f'Path Index: {path_idx}\n')
            #                 f.write('Coefficients:\n')
            #                 np.savetxt(f, path_coeffs_image.cpu().numpy(), fmt='%f')


            # if (is_visualize) and (page_idx % 1024 == 0):
            #     # Create a subdirectory in the plot output directory for visualizations
            #     signal_plot_dir = os.path.join(wavelet_batch_plot_dir, f'visualize_1d_signal_[{effective_p_start},{effective_p_end}]_[{page_idx},{page_idx+N}]')
            #     os.makedirs(signal_plot_dir, exist_ok=True)

            #     # Call the existing visualize functions to save the plots
            #     visualize_wavelet_1d(coeffs, page_idx, signal_plot_dir)
            #     create_scalogram_1d(wavelet, block, page_idx, signal_plot_dir)
            
            # print(f"Processed wavelet for block [{t_start}, {t_start}] X [{effective_p_start},{effective_p_end}]_[{page_idx},{page_idx+N}]")



def process_2d_parallel_scattering_wavelets_with_scalogram(__npy_only, row_initial, combined_data, n, t, output_data_plot_dirs, is_complex, is_visualize, wavelet, K=8, J=4, L=8):
    """
    Processes blocks of the combined data using 2D Wavelet Scattering, visualizes and saves the coefficients, and generates scalograms.

    Parameters:
    combined_data (np.ndarray): The 2D array where rows are pages and columns are time.
    n (int): Number of pages per block.
    t (int): Divisor for the time axis, to determine the width of the block in time.
    output_data_plot_dirs (list): Directories to save the output data and plots.
    J (int): Number of scales for the 2D scattering. Default is 4.
    L (int): Number of angles for the 2D scattering. Default is 8.
    """ 
    print("process_2d_parallel_scattering_wavelets_with_scalogram")
    # Separate the output for data and plots
    output_data_dir, output_plot_dir = output_data_plot_dirs

    num_pages, num_time_points = combined_data.shape
    block_time_size = num_time_points // t

    start = 0 if row_initial is None else row_initial

    # Define the root directories based on wavelet type
    wavelet_root_data_dir = os.path.join(output_data_dir, f"{wavelet}_wavelet_{'complex' if is_complex else 'real'}")
    wavelet_root_plot_dir = os.path.join(output_plot_dir, f"{wavelet}_wavelet_{'complex' if is_complex else 'real'}")

    # Ensure the directories exist
    os.makedirs(wavelet_root_data_dir, exist_ok=True)
    os.makedirs(wavelet_root_plot_dir, exist_ok=True)
    

    for batch in range (0, t):
        # Define the current batch
        wavelet_batch_data_dir = os.path.join(wavelet_root_data_dir, f'batch_{batch}')
        wavelet_batch_plot_dir = os.path.join(wavelet_root_plot_dir, f'batch_{batch}')

        # Ensure the directories exist
        os.makedirs(wavelet_batch_data_dir, exist_ok=True)
        os.makedirs(wavelet_batch_plot_dir, exist_ok=True)

        # Define the edges of samples of the current batch
        t_start = block_time_size * batch
        t_end = t_start + block_time_size - 1

        signal_indices = list(range(start, num_pages, n))

        for signal_index in signal_indices:
            # Define the edges of samples of the current batch
            p_start = signal_index
            p_end = p_start + n - 1
            print('startt ', t_start, ' endt ', t_end )
            print('startp ', p_start, ' endp ', p_end )

            signal = combined_data[p_start:(p_end+1), t_start:(t_end+1)]

            # Divide this chunk into K parts for K threads
            signal_parts = np.array_split(signal, K)
            barrier = threading.Barrier(K)  # Synchronize threads after wavelet computation
            order_lock = threading.Lock()   # Lock for ensuring ordered writing

            # Launch K threads for processing the signal parts
            with ThreadPoolExecutor(max_workers=K) as executor:
                for thread_id, signal_part in enumerate(signal_parts):
                    # Calculate the section directory based on thread-specific part
                    thread_p_start = p_start + thread_id * (n // K)
                    thread_p_end = thread_p_start + (n // K) - 1
                    executor.submit(process_wavelet_thread_part, __npy_only, thread_id, signal_part, t_start, t_end, wavelet_batch_data_dir, wavelet, J, barrier, K, order_lock, thread_p_start, thread_p_end)


def process_wavelet_thread_part(__npy_only, thread_id, signal_part, t_start, t_end, wavelet_batch_data_dir, wavelet, J, barrier, total_threads, order_lock, p_start, p_end) :
        # Separate real and imaginary parts
        real_signal = np.real(signal_part)
        imag_signal = np.imag(signal_part)
        print(thread_id, " ----> START")
        # Convert both parts to tensors
        real_tensor = torch.from_numpy(real_signal).float().unsqueeze(0).unsqueeze(0)
        imag_tensor = torch.from_numpy(imag_signal).float().unsqueeze(0).unsqueeze(0)

        # Perform 2D wavelet scattering on the block
        scattering = Scattering2D(J=J, shape=signal_part.shape)  # Initialize the scattering object with the block shape
            
        # Process both parts through scattering
        coeffs_real = scattering(real_tensor)
        coeffs_imag = scattering(imag_tensor)

        print("thread_id: ", thread_id, ": T from ", t_start, " to ", t_end, " --- P from ", p_start, " to ", p_end)
        print(coeffs_real.shape)
        print(coeffs_real)
        print('---------------------------------------------')
        print(coeffs_imag.shape)
        print(coeffs_imag)

        # Combine the real and imaginary coefficients if needed
        coeffs = (coeffs_real, coeffs_imag)
        # Wait until all threads finish wavelet computation
        barrier.wait()

        # Lock and ensure ordered writing based on thread_id
        with order_lock:
            # Create the directory for this batch of signals under the wavelet-specific directory
            batch_dir = os.path.join(wavelet_batch_data_dir, f'{wavelet}_2d_signal_{p_start}_{p_end}')
            os.makedirs(batch_dir, exist_ok=True)

            # Create directories for storing .npy and .txt files
            npy_dir = os.path.join(batch_dir, 'npy')
            text_dir = os.path.join(batch_dir, 'text')
            os.makedirs(npy_dir, exist_ok=True)
            os.makedirs(text_dir, exist_ok=True)

            # Create directories for storing real and image files
            npy_dir_real = os.path.join(npy_dir, 'real')
            npy_dir_imag = os.path.join(npy_dir, 'image')
            os.makedirs(npy_dir_real, exist_ok=True)
            os.makedirs(npy_dir_imag, exist_ok=True)

            txt_dir_real = os.path.join(text_dir, 'real')
            txt_dir_imag = os.path.join(text_dir, 'image')
            os.makedirs(txt_dir_real, exist_ok=True)
            os.makedirs(txt_dir_imag, exist_ok=True)

            # Loop over each scattering path of real data
            for path_idx in range(coeffs_real.shape[2]):  # Loop over scattering paths
                # Extract coefficients for the current path
                path_coeffs_real = coeffs_real[0, 0, path_idx, :, :]  # Shape: (height, width), assuming batch size is 1

                print("saving npy real")
                # Save real part as .npy
                np.save(os.path.join(npy_dir_real, f'scattering_coeffs_path_{path_idx}_real.npy'), path_coeffs_real.cpu().numpy())
                
                if not __npy_only:
                    print("saving text real")
                    # Save real part in human-readable .txt format
                    with open(os.path.join(txt_dir_real, f'scattering_coeffs_path_{path_idx}_real.txt'), 'w') as f:
                        f.write(f'===== Scattering Coefficients (Real Part) for Path {path_idx} =====\n')
                        f.write(f'Coefficient dimensions: {path_coeffs_real.shape}\n')
                        f.write(f'Path Index: {path_idx}\n')
                        f.write('Coefficients:\n')
                        np.savetxt(f, path_coeffs_real.cpu().numpy(), fmt='%f')
            
            if coeffs_imag is not None:
                # Loop over each scattering path of imaginary data
                for path_idx in range(coeffs_imag.shape[2]):  # Loop over scattering paths
                    # Extract coefficients for the current path
                    path_coeffs_image = coeffs_imag[0, 0, path_idx, :, :]  # Shape: (height, width), assuming batch size is 1

                    print("saving npy real")
                    np.save(os.path.join(npy_dir_imag, f'scattering_coeffs_path_{path_idx}_imag.npy'), path_coeffs_image.cpu().numpy())
                    if not __npy_only: 
                        # Save imaginary part in human-readable .txt format
                        print("saving text real")
                        with open(os.path.join(txt_dir_imag, f'scattering_coeffs_path_{path_idx}_imag.txt'), 'w') as f:
                            f.write(f'===== Scattering Coefficients (Imaginary Part) for Path {path_idx} =====\n')
                            f.write(f'Coefficient dimensions: {path_coeffs_image.shape}\n')
                            f.write(f'Path Index: {path_idx}\n')
                            f.write('Coefficients:\n')
                            np.savetxt(f, path_coeffs_image.cpu().numpy(), fmt='%f')


            # if (is_visualize) and (signal_index % 1024 == 0):
            #     # Create a subdirectory in the plot output directory for visualizations
            #     signal_plot_dir = os.path.join(wavelet_batch_plot_dir, f'visualize_1d_signal_{p_start}_{p_end}')
            #     os.makedirs(signal_plot_dir, exist_ok=True)

            #     # Call the existing visualize functions to save the plots
            #     visualize_wavelet_1d(coeffs, signal_index, signal_plot_dir)
            #     create_scalogram_1d(wavelet, signal, signal_index, signal_plot_dir)
            
            # print(f"Processed wavelet for block [{t_start}, {t_start}] X [{p_start}, {p_end}]")



def main():
    parser = argparse.ArgumentParser(description="Choose the type of analysis.")
    #parser.add_argument('--help', action='store_true', help="Prints the flags with explanations and gives examples.")
    parser.add_argument('--combine', action='store_true', help="Combines the Hamming and Cosine data into a single file.")
    parser.add_argument('--analysis', choices=['wavelet', 'wavelet1d', 'wavelet2d', 'wavelet1d_batches'], required=True, help="Type of analysis to perform.")
    parser.add_argument('--test-folder', required=True, help="Name of the test folder inside 'results/1/'")
    parser.add_argument('--analysis-type', required=True, help="Type of spectrum analysis, e.g. KL'")
    parser.add_argument('--pre-combined', action='store_true', help="Use pre-combined data if available.")
    parser.add_argument('--is-complex', action='store_true', help="Handle the data as complex during wavelet analysis.")
    parser.add_argument('--start-row', type=int, help="Handle the data as complex during wavelet analysis.")
    
    parser.add_argument('--organize', action='store_true', help="Organize folders")

    # Parse known arguments first to capture --analysis
    args, unknown = parser.parse_known_args()

    if (args.analysis == 'wavelet1d_batches') or (args.analysis == 'wavelet2d'):
        parser.add_argument('--batches', type=int, required=True, help="Number of batches if specified in --analysis")
        parser.add_argument('--visualize', action='store_true', help="Visualizes the scalogram and the coeffs, mag and phase.")
        parser.add_argument('--exp-decay', action='store_true', help="creates a decaying overlap between adjacent blocks.")
        parser.add_argument('--scattering', action='store_true', help="Perform Wavelet Scattering Algorithm")
        parser.add_argument('--parallel', action='store_true', help="creates a parallel 2D wavelet calculation.")
        parser.add_argument('--num-threads', type=int, help="creates a parallel 2D wavelet calculation.")
        parser.add_argument('--npy-only', action='store_true', help="Store only Numpy files, no Text files - to save space")
        parser.add_argument('--num-pages', type=int, required=True, help="Number of samples in block if specified in --analysis")
        parser.add_argument('--wavelet', type=str, required=True, help="wavelet to use")
    args = parser.parse_args()

    if len(vars(args)) == 0:
        print_custom_help()
        return
    
    # base_dir = r'C:\thesis\results\1'
    base_dir = '/Volumes/Extreme SSD/thesis/runs/mathematical'

    # Directories containing the Hamming and Cosine distance files
    hamming_directory = os.path.join(base_dir, args.test_folder, 'hamming')
    cosine_directory = os.path.join(base_dir, args.test_folder, 'cosine')

    combined_data_path = os.path.join(base_dir, args.test_folder, 'combined_data.npy')
    filtered_data_path = os.path.join(base_dir, args.test_folder, 'filtered_combined_data.npy')
    bitmap_path = os.path.join(base_dir, args.test_folder, 'bitmap.txt')

    if args.combine:
        if args.test_folder:
            combine_data(hamming_directory, cosine_directory, combined_data_path, filtered_data_path, bitmap_path)
        else:
            print("Error: --test-folder is required for --combine.")
        return
    
    if args.pre_combined and os.path.exists(combined_data_path) and os.path.exists(filtered_data_path) and os.path.exists(bitmap_path):
        combined_data = np.load(filtered_data_path)
        print("Loaded pre-combined data.")
    else:
        combine_data(hamming_directory, cosine_directory, combined_data_path, filtered_data_path, bitmap_path)

    # Parameters
    window_size = 32  # Size of the moving window
    step_size = 16    # Step size, ensuring 50% overlap
    num_bins = 10     # Number of bins for histogram

    # Directory to save the images
    output_dir_root =  os.path.join(base_dir, args.test_folder, args.analysis_type)

    output_dir = os.path.join(output_dir_root, f'n{args.num_pages}t{args.batches}')

    output_data_dir = os.path.join(output_dir, 'data')
    output_plot_dir = os.path.join(output_dir, 'plots')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_data_dir, exist_ok=True)
    os.makedirs(output_plot_dir, exist_ok=True)

    output_data_plot_dirs = [output_data_dir, output_plot_dir]

    # Check if --is-complex exists and is greater than zero
    if args.start_row is not None:
        if args.start_row > 0:
            row_initial = args.start_row
        else:
            parser.error("--is-complex must be an integer greater than zero.")
    else:
        row_initial = None  # Or handle the case where it's not provided

    # You can now use row_initial in your script
    print(f"row_initial: {row_initial}")

    if args.analysis == 'wavelet1d_batches':
        process_1d_wavelet_time_segment(args.batches, row_initial, combined_data, output_data_plot_dirs, wavelet='db4', level=2, is_complex=True)
    elif args.analysis == 'wavelet1d':
        process_1d_wavelet(row_initial, combined_data, output_data_plot_dirs, wavelet='db4', level=2, is_complex=True)
    elif args.analysis == 'wavelet2d':
        n = args.num_pages  # Number of pages per block
        t = args.batches   # Divisor for the time axis to create block width
        visualize = args.visualize
        __npy_only = args.npy_only
        if args.exp_decay:
            if args.scattering:
                if args.parallel:
                    num_of_threads = args.num_threads
                    process_2d_scattering_parallel_overlap(__npy_only, visualize, row_initial, combined_data, n, t, output_data_plot_dirs, args.is_complex, 'db4', J=4, L=8)

                else:
                    process_2d_scattering_wavelets_with_scalogram_overlap(__npy_only, visualize, row_initial, combined_data, n//4, t,
                                                                            output_data_plot_dirs, args.is_complex, wavelet='db4', J=4, L=8, 
                                                                            page_overlap_percentage=0.5, page_decay_rate=0.1, 
                                                                            batch_overlap_percentage=0.5)
            else:
                if args.parallel:
                    process_block_2d_wavelet_features_overlap_parallel(combined_data, row_initial, n, t, output_data_plot_dirs, 
                                        args.wavelet, level=3, is_visualize=False, is_complex=False)
                else:
                    process_block_2d_wavelet_features_overlap(combined_data, row_initial, n, t, output_data_plot_dirs, 
                                        args.wavelet, level=3, is_visualize=False, is_complex=False)

        else:
            if args.scattering:
                if args.parallel:
                    num_of_threads = args.num_threads
                    process_2d_parallel_scattering_wavelets_with_scalogram(__npy_only, row_initial, combined_data, n, t, output_data_plot_dirs, args.is_complex, visualize, 'db4', num_of_threads, J=4, L=8)
                else:
                    process_2d_scattering_wavelets_with_scalogram(__npy_only, row_initial, combined_data, n, t, output_data_plot_dirs, args.is_complex, visualize, wavelet='db4', J=4, L=8)
            else:
                if args.parallel:
                    process_block_2d_wavelet_features_parallel(combined_data, row_initial, n, t, output_data_plot_dirs, 
                                      args.wavelet, level=3, is_visualize=False, is_complex=False)
                else:
                    process_block_2d_wavelet_features(combined_data, row_initial, n, t, output_data_plot_dirs, 
                                      args.wavelet, level=3, is_visualize=False, is_complex=False)
                

    print('Done')


if __name__ == "__main__":
    main()