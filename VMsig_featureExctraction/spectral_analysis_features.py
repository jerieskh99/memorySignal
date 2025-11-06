import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import h5py 
from matplotlib.colors import LogNorm
from scipy.signal import find_peaks
from concurrent.futures import ProcessPoolExecutor

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


##########################################
#          Fourier 2D - Features         #
##########################################

def visualize_real_imaginary_fft(fourier_block, fft_batch_plot_dir, effective_p_start, effective_p_end, t_start_idx, t_end_idx):
    """
    Visualizes the real and imaginary parts of the Fourier-transformed data in one PNG file.

    Parameters:
    fourier_block (numpy.ndarray): The 2D Fourier-transformed block (complex data).
    fft_batch_plot_dir (str): Directory where the plot will be saved.
    effective_p_start (int): Starting page number for naming the file.
    effective_p_end (int): Ending page number for naming the file.
    t_start_idx (int): Start time index.
    t_end_idx (int): End time index.
    """
    # Extract real and imaginary parts
    fourier_real = np.real(fourier_block)
    fourier_imag = np.imag(fourier_block)
    
    # Set vmin to avoid log(0) and create better contrast
    vmin_real = np.min(np.abs(fourier_real[np.abs(fourier_real) > 0]))
    vmin_imag = np.min(np.abs(fourier_imag[np.abs(fourier_imag) > 0]))

    # Create a figure with two subplots: Real and Imaginary
    plt.figure(figsize=(12, 8))
    
    # Plot the real part
    plt.subplot(2, 1, 1)
    plt.imshow(np.abs(fourier_real), aspect='auto', cmap='coolwarm', origin='lower', norm=LogNorm(vmin=vmin_real, vmax=np.max(np.abs(fourier_real))))
    plt.title('Log-Normalized Real Part of Fourier Transform')
    plt.xlabel('Frequency')
    plt.ylabel('Page')
    plt.colorbar(label='Amplitude')
    
    # Plot the imaginary part
    plt.subplot(2, 1, 2)
    plt.imshow(np.abs(fourier_imag), aspect='auto', cmap='coolwarm', origin='lower', norm=LogNorm(vmin=vmin_imag, vmax=np.max(np.abs(fourier_imag))))
    plt.title('Log-Normalized Imaginary Part of Fourier Transform')
    plt.xlabel('Frequency')
    plt.ylabel('Page')
    plt.colorbar(label='Amplitude')

    # Save the figure
    plt.tight_layout()
    plot_filename = os.path.join(fft_batch_plot_dir, f"{effective_p_start}_{effective_p_end}_{t_start_idx}_{t_end_idx}_real_imaginary.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close()


def visualize_magnitude_phase_fft(fourier_block, fft_batch_plot_dir, effective_p_start, effective_p_end, t_start_idx, t_end_idx):
    """
    Visualizes the magnitude and phase of the Fourier-transformed data in one PNG file.

    Parameters:
    fourier_block (numpy.ndarray): The 2D Fourier-transformed block (complex data).
    fft_batch_plot_dir (str): Directory where the plot will be saved.
    effective_p_start (int): Starting page number for naming the file.
    effective_p_end (int): Ending page number for naming the file.
    t_start_idx (int): Start time index.
    t_end_idx (int): End time index.
    """
    # Calculate magnitude and phase
    fourier_magnitude = np.abs(fourier_block)
    fourier_phase = np.angle(fourier_block)
    
    # Set vmin to avoid log(0) and create better contrast
    vmin_mag = np.min(np.abs(fourier_magnitude[np.abs(fourier_magnitude) > 0]))
    vmin_phase = np.min(np.abs(fourier_phase[np.abs(fourier_phase) > 0]))

    # Create a figure with two subplots: Magnitude and Phase
    plt.figure(figsize=(12, 8))
    
    # Plot the magnitude with LogNorm
    plt.subplot(2, 1, 1)
    plt.imshow(fourier_magnitude, aspect='auto', cmap='inferno', origin='lower', norm=LogNorm(vmin=vmin_mag, vmax=np.max(fourier_magnitude)))
    plt.title('Log-Normalized Magnitude of Fourier Transform')
    plt.xlabel('Frequency')
    plt.ylabel('Page')
    plt.colorbar(label='Magnitude')

    # Plot the phase without LogNorm (phase is better visualized linearly)
    plt.subplot(2, 1, 2)
    plt.imshow(fourier_phase, aspect='auto', cmap='twilight', origin='lower')
    plt.title('Phase of Fourier Transform')
    plt.xlabel('Frequency')
    plt.ylabel('Page')
    plt.colorbar(label='Phase (radians)')

    # Save the figure
    plt.tight_layout()
    plot_filename = os.path.join(fft_batch_plot_dir, f"{effective_p_start}_{effective_p_end}_{t_start_idx}_{t_end_idx}_magnitude_phase.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close()


def visualize_fft(fft_block, fft_batch_plot_dir, effective_p_start, effective_p_end, t_start_idx, t_end_idx):
    """
    Visualizes both the magnitude and phase, as well as real and imaginary parts of the Fourier-transformed data.

    Parameters:
    fourier_block (numpy.ndarray): The 2D Fourier-transformed block (complex data).
    fft_batch_plot_dir (str): Directory where the plot will be saved.
    effective_p_start (int): Starting page number for naming the file.
    effective_p_end (int): Ending page number for naming the file.
    t_start_idx (int): Start time index.
    t_end_idx (int): End time index.
    """
    # Visualize the real and imaginary parts
    visualize_real_imaginary_fft(fft_block, fft_batch_plot_dir, effective_p_start, effective_p_end, t_start_idx, t_end_idx)
    
    # Visualize the magnitude and phase
    visualize_magnitude_phase_fft(fft_block, fft_batch_plot_dir, effective_p_start, effective_p_end, t_start_idx, t_end_idx)


def fft_create_hdf5_2d(data, data_dir, effective_p_start, effective_p_end, t_start, t_end):
    hdf5_file = os.path.join(data_dir, f'{effective_p_start}_{effective_p_end}_{t_start}_{t_end}_fft.h5')
    
    # Calculate magnitude and phase
    magnitude = np.abs(data)
    phase = np.angle(data)

    # Create and open the HDF5 file
    with h5py.File(hdf5_file, 'w') as h5file:
        h5file.create_dataset('magnitude', data=magnitude, compression="gzip")
        h5file.create_dataset('phase', data=phase, compression="gzip")

    print(f"Fourier magnitude and phase data successfully saved to {hdf5_file}")


def fft_create_hdf5_2d_features(features, features_dir, effective_p_start, effective_p_end, t_start, t_end):
    # Define the file name based on the input parameters
    hdf5_file = os.path.join(features_dir, f'{effective_p_start}_{effective_p_end}_{t_start}_{t_end}_features.h5')
    
    # Create and open the HDF5 file
    with h5py.File(hdf5_file, 'w') as h5file:
        # Iterate through the features and store each in the HDF5 file
        for feature_name, value in features.items():
            # If the value is a scalar, store it as a scalar dataset
            if np.isscalar(value):
                h5file.create_dataset(feature_name, data=value)
            else:
                # If the value is an array, store it as a dataset
                h5file.create_dataset(feature_name, data=value, compression="gzip")
    
    print(f"Features successfully saved to {hdf5_file}")


def fft_create_files(data, features, data_dir, features_dir, effective_p_start ,effective_p_end, t_start, t_end):
    fft_create_hdf5_2d(data, data_dir, effective_p_start ,effective_p_end, t_start, t_end)
    fft_create_hdf5_2d_features(features, features_dir, effective_p_start ,effective_p_end, t_start, t_end)


def extract_fft_features(fft_block):
    """
    Extracts features from a block of FFT-transformed data.
    
    Parameters:
    fft_block (numpy.ndarray): 2D array of Fourier-transformed data (complex).
    
    Returns:
    dict: A dictionary containing the extracted features.
    """
    
    # Initialize feature dictionary
    features = {}
    
    # Energy
    energy = np.sum(np.abs(fft_block)**2)
    features['energy'] = energy
    
    # Power Spectral Density (PSD)
    psd = np.abs(fft_block)**2 / fft_block.size
    features['psd_mean'] = np.mean(psd)
    features['psd_variance'] = np.var(psd)
    
    # Dominant Frequencies (taking the top 3 frequencies as an example)
    dominant_freqs = np.argsort(np.abs(fft_block).flatten())[-3:]
    features['dominant_freqs'] = dominant_freqs
    
    # Spectral Entropy
    prob_dist = psd / np.sum(psd)
    spectral_entropy = -np.sum(prob_dist * np.log2(prob_dist + 1e-10))  # small epsilon to avoid log(0)
    features['spectral_entropy'] = spectral_entropy
    
    # Spectral Centroid
    freqs = np.fft.fftfreq(fft_block.shape[1])  # X-axis assumed as frequency dimension
    spectral_centroid = np.sum(freqs * np.abs(fft_block).mean(axis=0)) / np.sum(np.abs(fft_block))
    features['spectral_centroid'] = spectral_centroid
    
    # Spectral Bandwidth
    spectral_bandwidth = np.sqrt(np.sum((freqs - spectral_centroid)**2 * np.abs(fft_block).mean(axis=0)) / np.sum(np.abs(fft_block)))
    features['spectral_bandwidth'] = spectral_bandwidth
    
    # Phase Information
    phase = np.angle(fft_block)
    phase_mean = np.mean(phase)
    phase_variance = np.var(phase)
    features['phase_mean'] = phase_mean
    features['phase_variance'] = phase_variance
    
    # Peak-to-Average Ratio
    peak_to_average = np.max(np.abs(fft_block)) / np.mean(np.abs(fft_block))
    features['peak_to_average'] = peak_to_average
    
    # Coefficient of Variation
    coeff_variation = np.std(np.abs(fft_block)) / np.mean(np.abs(fft_block))
    features['coeff_variation'] = coeff_variation
    
    return features


def process_block_2d_stft_decay_features_single_batch(full_signal, row_initial, n, batch_index, t_start_idx_flt, t_end_idx__flt, num_pages, output_data_plot_dirs
                                                      , is_visualize, page_overlap_percentage, page_decay_rate):
    t_start_idx = int(t_start_idx_flt)
    t_end_idx = int(t_end_idx__flt)
    overlap_pages = int(n * page_overlap_percentage)
    start = 0 if row_initial is None else row_initial

    # Separate the output for data and plots
    output_data_dir, output_plot_dir, output_feature_dir = output_data_plot_dirs


    # Define the current batch directories
    fft_batch_data_dir = os.path.join(output_data_dir, f'batch_{batch_index}')
    fft_batch_plot_dir = os.path.join(output_plot_dir, f'batch_{batch_index}')
    fft_batch_features_dir = os.path.join(output_feature_dir, f'batch_{batch_index}')

    # Ensure the directories exist
    os.makedirs(fft_batch_data_dir, exist_ok=True)
    os.makedirs(fft_batch_plot_dir, exist_ok=True)
    os.makedirs(fft_batch_features_dir, exist_ok=True)

    # Function to wrap around the indices in the time or page axis
    def wrap_index(index, max_value):
        """ Returns a wrapped index to handle negative or out-of-bounds indices. """
        return index % max_value
    
    # Random selection for visualization
    if is_visualize:
        mid_page = num_pages // 2
        # Randomly select one page from the first half and one from the second half
        random_page_1st_half = np.random.randint(start, mid_page - n)
        random_page_2nd_half = np.random.randint(mid_page, num_pages - n)

    for page_idx in range(start, num_pages, n):
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
        fft_block = np.fft.fft2(block)

        if is_visualize and (
            (effective_p_start <= random_page_1st_half < effective_p_end) or
            (effective_p_start <= random_page_2nd_half < effective_p_end)
        ):
            visualize_fft(fft_block, fft_batch_plot_dir, effective_p_start, effective_p_end, t_start_idx, t_end_idx)            

        # Exctract all possible features
        features = extract_fft_features(fft_block)

        fft_create_files(fft_block, features, fft_batch_data_dir, fft_batch_features_dir, effective_p_start, effective_p_end, t_start_idx, t_end_idx)

        print(f"Processed FFT for block [{t_start_idx}, {t_end_idx}] X [{effective_p_start}, {effective_p_end}]")
    
    


def process_block_2d_stft_features_overlap_parallel(full_signal, row_initial, n, num_batches, output_data_plot_dirs, 
                                                    is_visualize=False, page_overlap_percentage=0.5,
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

    with ProcessPoolExecutor(max_workers=num_batches) as executer:
        futures = [
            executer.submit(
                process_block_2d_stft_decay_features_single_batch, full_signal, row_initial, n, batch_index, t_start_index, t_end_index, 
                num_pages, output_data_plot_dirs, is_visualize, 
                page_overlap_percentage, page_decay_rate
            )
            for batch_index, t_start_index, t_end_index in index_map
        ]

        # Wait for all futures to complete
        for future in futures:
            future.result()


##########################################
#         Cepstrum 2D - Features         #
##########################################

def normalize_data(data):
    """Normalize the data to a 0-1 range."""
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max - data_min == 0:
        return data  # Prevent division by zero if all values are the same
    return (data - data_min) / (data_max - data_min)


def visualize_real_imaginary_cepstrum(cepstrum_block, fft_batch_plot_dir, effective_p_start, effective_p_end, t_start_idx, t_end_idx):
    """
    Visualizes the real and imaginary parts of the cepstrum in one PNG file.

    Parameters:
    cepstrum_block (numpy.ndarray): The 2D cepstrum block (complex data).
    fft_batch_plot_dir (str): Directory where the plot will be saved.
    effective_p_start (int): Starting page number for naming the file.
    effective_p_end (int): Ending page number for naming the file.
    t_start_idx (int): Start time index.
    t_end_idx (int): End time index.
    """
    # Extract real and imaginary parts
    cepstrum_real = np.real(cepstrum_block)
    cepstrum_imag = np.imag(cepstrum_block)
    
    # Set vmin to avoid log(0) and create better contrast
    vmin_real = np.min(np.abs(cepstrum_real[np.abs(cepstrum_real) > 0]))
    vmin_imag = np.min(np.abs(cepstrum_imag[np.abs(cepstrum_imag) > 0]))

    # Create a figure with two subplots: Real and Imaginary
    plt.figure(figsize=(12, 8))
    
    # Plot the real part
    plt.subplot(2, 1, 1)
    plt.imshow(np.abs(cepstrum_real), aspect='auto', cmap='coolwarm', origin='lower', norm=LogNorm(vmin=vmin_real, vmax=np.max(np.abs(cepstrum_real))))
    plt.title('Log-Normalized Real Cepstrum')
    plt.xlabel('Quefrency (samples)')
    plt.ylabel('Page')
    plt.colorbar(label='Amplitude')
    
    # Plot the imaginary part
    plt.subplot(2, 1, 2)
    plt.imshow(np.abs(cepstrum_imag), aspect='auto', cmap='coolwarm', origin='lower', norm=LogNorm(vmin=vmin_imag, vmax=np.max(np.abs(cepstrum_imag))))
    plt.title('Log-Normalized Imaginary Cepstrum')
    plt.xlabel('Quefrency (samples)')
    plt.ylabel('Page')
    plt.colorbar(label='Amplitude')

    # Save the figure
    plt.tight_layout()
    plot_filename = os.path.join(fft_batch_plot_dir, f"{effective_p_start}_{effective_p_end}_{t_start_idx}_{t_end_idx}_real_imaginary.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close()


def visualize_magnitude_phase_cepstrum(cepstrum_block, fft_batch_plot_dir, effective_p_start, effective_p_end, t_start_idx, t_end_idx):
    """
    Visualizes the magnitude and phase of the cepstrum in one PNG file.

    Parameters:
    cepstrum_block (numpy.ndarray): The 2D cepstrum block (complex data).
    fft_batch_plot_dir (str): Directory where the plot will be saved.
    effective_p_start (int): Starting page number for naming the file.
    effective_p_end (int): Ending page number for naming the file.
    t_start_idx (int): Start time index.
    t_end_idx (int): End time index.
    """
    # Calculate magnitude and phase
    cepstrum_magnitude = np.abs(cepstrum_block)
    cepstrum_phase = np.angle(cepstrum_block)
    
    # Set vmin to avoid log(0) and create better contrast
    vmin_mag = np.min(np.abs(cepstrum_magnitude[np.abs(cepstrum_magnitude) > 0]))
    vmin_phase = np.min(np.abs(cepstrum_phase[np.abs(cepstrum_phase) > 0]))

    # Create a figure with two subplots: Magnitude and Phase
    plt.figure(figsize=(12, 8))
    
    # Plot the magnitude with LogNorm
    plt.subplot(2, 1, 1)
    plt.imshow(cepstrum_magnitude, aspect='auto', cmap='inferno', origin='lower', norm=LogNorm(vmin=vmin_mag, vmax=np.max(cepstrum_magnitude)))
    plt.title('Log-Normalized Magnitude of Cepstrum')
    plt.xlabel('Quefrency (samples)')
    plt.ylabel('Page')
    plt.colorbar(label='Magnitude')

    # Plot the phase without LogNorm (phase is better visualized linearly)
    plt.subplot(2, 1, 2)
    plt.imshow(cepstrum_phase, aspect='auto', cmap='twilight', origin='lower')
    plt.title('Phase of Cepstrum')
    plt.xlabel('Quefrency (samples)')
    plt.ylabel('Page')
    plt.colorbar(label='Phase (radians)')

    # Save the figure
    plt.tight_layout()
    plot_filename = os.path.join(fft_batch_plot_dir, f"{effective_p_start}_{effective_p_end}_{t_start_idx}_{t_end_idx}_magnitude_phase.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close()


def visualize_cepstrum(cepstrum_block, fft_batch_plot_dir, effective_p_start, effective_p_end, t_start_idx, t_end_idx):
    """
    Visualizes both the magnitude and phase of the 2D cepstrum as separate heatmaps.
    
    Parameters:
    cepstrum_block (numpy.ndarray): The 2D cepstrum block to visualize.
    fft_batch_plot_dir (str): The directory where the plot will be saved.
    effective_p_start (int): Starting page number for naming the file.
    effective_p_end (int): Ending page number for naming the file.
    t_start_idx (int): Start time index.
    t_end_idx (int): End time index.
    """
    # normalized_cepstrum = normalize_data(cepstrum_block)
    visualize_real_imaginary_cepstrum(cepstrum_block, fft_batch_plot_dir, effective_p_start, effective_p_end, t_start_idx, t_end_idx)
    visualize_magnitude_phase_cepstrum(cepstrum_block, fft_batch_plot_dir, effective_p_start, effective_p_end, t_start_idx, t_end_idx)


def cepstrum_create_hdf5_2d(data, data_dir, effective_p_start, effective_p_end, t_start, t_end):
    hdf5_file = os.path.join(data_dir, f'{effective_p_start}_{effective_p_end}_{t_start}_{t_end}_cepstrum.h5')
    
    # Calculate magnitude and phase
    real_cep = np.real(data)
    imag_cep = np.imag(data)

    # Create and open the HDF5 file
    with h5py.File(hdf5_file, 'w') as h5file:
        h5file.create_dataset('real', data=real_cep, compression="gzip")
        h5file.create_dataset('image', data=imag_cep, compression="gzip")

    print(f"Fourier magnitude and phase data successfully saved to {hdf5_file}")


def cepstrum_create_hdf5_2d_features(features, features_dir, effective_p_start, effective_p_end, t_start, t_end):
    # Define the file name based on the input parameters
    hdf5_file = os.path.join(features_dir, f'{effective_p_start}_{effective_p_end}_{t_start}_{t_end}_cepstrum_features.h5')
    
    # Create and open the HDF5 file
    with h5py.File(hdf5_file, 'w') as h5file:
        # Iterate through the features and store each in the HDF5 file
        for feature_name, value in features.items():
            # If the value is a scalar, store it as a scalar dataset
            if np.isscalar(value):
                h5file.create_dataset(feature_name, data=value)
            else:
                # If the value is an array, store it as a dataset
                h5file.create_dataset(feature_name, data=value, compression="gzip")
    
    print(f"Features successfully saved to {hdf5_file}")


def cepstrum_create_files(data, features, data_dir, features_dir, effective_p_start ,effective_p_end, t_start, t_end):
    cepstrum_create_hdf5_2d(data, data_dir, effective_p_start ,effective_p_end, t_start, t_end)
    cepstrum_create_hdf5_2d_features(features, features_dir, effective_p_start ,effective_p_end, t_start, t_end)


def extract_cepstrum_features(cepstrum_block):
    """
    Extract features from a cepstrum block.
    
    Parameters:
    cepstrum_block (numpy.ndarray): The real part of a cepstrum (output from cepstrum calculation).
    
    Returns:
    dict: A dictionary containing the extracted features from the cepstrum block.
    """
    
    # Initialize the dictionary to store features
    features = {}
    
    # Energy (sum of squared magnitudes)
    energy = np.sum(np.abs(cepstrum_block)**2)
    features['energy'] = energy
    
    # Mean and Variance of Cepstrum
    cepstrum_mean = np.mean(cepstrum_block)
    cepstrum_variance = np.var(cepstrum_block)
    features['cepstrum_mean'] = cepstrum_mean
    features['cepstrum_variance'] = cepstrum_variance
    
    # Spectral Entropy (entropy of the cepstrum block)
    prob_dist = np.abs(cepstrum_block)**2 / np.sum(np.abs(cepstrum_block)**2)
    spectral_entropy = -np.sum(prob_dist * np.log2(prob_dist + 1e-10))  # Avoid log(0) with small epsilon
    features['spectral_entropy'] = spectral_entropy
    
    # Peak-to-Average Ratio
    peak_to_average = np.max(np.abs(cepstrum_block)) / np.mean(np.abs(cepstrum_block))
    features['peak_to_average'] = peak_to_average
    
    # Dominant Quefrency (ignoring quefrency 0)
    cepstrum_without_zero = np.abs(cepstrum_block[1:])  # Ignore the 0th quefrency
    dominant_quefrency = np.argmax(cepstrum_without_zero) + 1  # Adjust index due to slicing
    features['dominant_quefrency'] = dominant_quefrency
    
    # Cepstrum Skewness (third central moment)
    skewness = np.mean((cepstrum_block - cepstrum_mean)**3) / (np.std(cepstrum_block)**3 + 1e-10)
    features['skewness'] = skewness
    
    # Cepstrum Kurtosis (fourth central moment)
    kurtosis = np.mean((cepstrum_block - cepstrum_mean)**4) / (np.var(cepstrum_block)**2 + 1e-10)
    features['kurtosis'] = kurtosis
    
    return features


def process_block_2d_cepstrum_decay_features_single_batch(full_signal, row_initial, n, batch_index, t_start_idx_flt, t_end_idx__flt, num_pages, output_data_plot_dirs
                                                      , is_visualize, page_overlap_percentage, page_decay_rate):
    t_start_idx = int(t_start_idx_flt)
    t_end_idx = int(t_end_idx__flt)
    overlap_pages = int(n * page_overlap_percentage)
    start = 0 if row_initial is None else row_initial

    # Separate the output for data and plots
    output_data_dir, output_plot_dir, output_feature_dir = output_data_plot_dirs


    # Define the current batch directories
    fft_batch_data_dir = os.path.join(output_data_dir, f'batch_{batch_index}')
    fft_batch_plot_dir = os.path.join(output_plot_dir, f'batch_{batch_index}')
    fft_batch_features_dir = os.path.join(output_feature_dir, f'batch_{batch_index}')

    # Ensure the directories exist
    os.makedirs(fft_batch_data_dir, exist_ok=True)
    os.makedirs(fft_batch_plot_dir, exist_ok=True)
    os.makedirs(fft_batch_features_dir, exist_ok=True)

    # Function to wrap around the indices in the time or page axis
    def wrap_index(index, max_value):
        """ Returns a wrapped index to handle negative or out-of-bounds indices. """
        return index % max_value
    
    # Random selection for visualization
    if is_visualize:
        mid_page = num_pages // 2
        # Randomly select one page from the first half and one from the second half
        random_page_1st_half = np.random.randint(start, mid_page - n)
        random_page_2nd_half = np.random.randint(mid_page, num_pages - n)

    for page_idx in range(start, num_pages, n):
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
        fft_block = np.fft.fft2(block)

        # Calculate log-magnitude spectrum
        log_magnitude_block = np.log(np.abs(fft_block) + 1e-10)  # small value to avoid log(0)
    
        #Perform IFFT to get the cepstrum
        cepstrum_block = np.fft.ifft2(log_magnitude_block)

        if is_visualize and (
            (effective_p_start <= random_page_1st_half < effective_p_end) or
            (effective_p_start <= random_page_2nd_half < effective_p_end)
        ):
            visualize_cepstrum(cepstrum_block, fft_batch_plot_dir, effective_p_start, effective_p_end, t_start_idx, t_end_idx)            

        # Exctract all possible features
        features = extract_cepstrum_features(np.real(cepstrum_block))

        cepstrum_create_files(fft_block, features, fft_batch_data_dir, fft_batch_features_dir, effective_p_start, effective_p_end, t_start_idx, t_end_idx)

        print(f"Processed cepstrum for block [{t_start_idx}, {t_end_idx}] X [{effective_p_start}, {effective_p_end}]")
    
    
def process_block_2d_cepstrum_features_overlap_parallel(full_signal, row_initial, n, num_batches, output_data_plot_dirs, 
                                                    is_visualize=False, page_overlap_percentage=0.5,
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

    with ProcessPoolExecutor(max_workers=num_batches) as executer:
        futures = [
            executer.submit(
                process_block_2d_cepstrum_decay_features_single_batch, full_signal, row_initial, n, batch_index, t_start_index, t_end_index, 
                num_pages, output_data_plot_dirs, is_visualize, 
                page_overlap_percentage, page_decay_rate
            )
            for batch_index, t_start_index, t_end_index in index_map
        ]

        # Wait for all futures to complete
        for future in futures:
            future.result()


##########################################
#             Helper Function            #
##########################################
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


def save_analysis_data(data, filename):
    """Helper function to save numpy array to file."""
    np.save(filename, data)


##########################################
#              Main Function             #
##########################################

def main():
    parser = argparse.ArgumentParser(description="Choose the type of analysis.")

    parser.add_argument('--analysis', choices=['fft', 'cepstrum'], required=True, help="Type of analysis to perform.")
    parser.add_argument('--test-folder', required=True, help="Name of the test folder inside 'results/1/'")
    parser.add_argument('--start-row', type=int, help="Handle the data as complex during wavelet analysis.")
    parser.add_argument('--batches', type=int, required=True, help="Number of batches if specified in --analysis")
    parser.add_argument('--visualize', action='store_true', help="Visualizes the scalogram and the coeffs, mag and phase.")
    parser.add_argument('--num-pages', type=int, required=True, help="Number of samples in block if specified in --analysis")

    args = parser.parse_args()

    if len(vars(args)) == 0:
        print_custom_help()
        return
    
    base_dir = '/Volumes/Extreme SSD/thesis/runs/mathematical'
    # base_dir = '/Users/jeries/Desktop/n8192t10'

    # Directory to save the images
    output_dir_root =  os.path.join(base_dir, args.test_folder)

    if args.analysis == 'fft':
        output_dir = os.path.join(output_dir_root, f'FFT_n{args.num_pages}t{args.batches}')
    elif args.analysis == 'cepstrum':
        output_dir = os.path.join(output_dir_root, f'cepstrum_n{args.num_pages}t{args.batches}')
    else:
        raise Exception("Only --analysis type is fft or cepstrum")

    output_data_dir = os.path.join(output_dir, 'data')
    output_plot_dir = os.path.join(output_dir, 'plots')
    output_featrues_dir = os.path.join(output_dir, 'features')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_data_dir, exist_ok=True)
    os.makedirs(output_plot_dir, exist_ok=True)
    os.makedirs(output_featrues_dir, exist_ok=True)

    output_data_plot_dirs = [output_data_dir, output_plot_dir, output_featrues_dir]

    combined_data_path = os.path.join(base_dir, args.test_folder, 'combined_data.npy')
    filtered_data_path = os.path.join(base_dir, args.test_folder, 'filtered_combined_data.npy')
    bitmap_path = os.path.join(base_dir, args.test_folder, 'bitmap.txt')

    if os.path.exists(combined_data_path) and os.path.exists(filtered_data_path) and os.path.exists(bitmap_path):
        combined_data = np.load(filtered_data_path)
        print("Loaded pre-combined data.")
    else:
        raise Exception("Combine files first using: wavelet_scattering_features.py --combine")

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

    if args.analysis == 'fft':
        process_block_2d_stft_features_overlap_parallel(full_signal=combined_data, row_initial=row_initial, n=args.num_pages, num_batches=args.batches,
                                                        output_data_plot_dirs=output_data_plot_dirs, is_visualize=args.visualize)
    elif args.analysis == 'cepstrum':
        process_block_2d_cepstrum_features_overlap_parallel(full_signal=combined_data, row_initial=row_initial, n=args.num_pages, num_batches=args.batches,
                                                        output_data_plot_dirs=output_data_plot_dirs, is_visualize=args.visualize)
    print('Done')


if __name__ == "__main__":
    main()