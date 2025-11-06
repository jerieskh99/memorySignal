import os
import numpy as np
import subprocess
from scipy.stats import entropy
import argparse
import scipy.stats as stats
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert
from sklearn.metrics import mutual_info_score
from scipy.stats import norm
from scipy.signal import welch
import itertools



# ============================================================
#                         File Creation 
# ============================================================

def save_to_txt(feature_data, feature_type):
    return


def save_to_npy(feature_data, feature_type):
    return


def make_complex(mag,phase):
    c_coeffs = mag * np.exp(1j * 2 * np.pi * phase)
    return c_coeffs


def calc_abs_fourier(complex_coeffs):
    fft = np.abs(fft(complex_coeffs))
    return fft


# ============================================================
#                     Energy Based Features
# ============================================================

def calculate_energy(mag, phase):
    """
    Calculate energy from magnitude and phase using their complex representation.

    Args:
    mag (np.array): Magnitude coefficients.
    phase (np.array): Phase coefficients.

    Returns:
    float: Energy of the combined complex representation of magnitude and phase.
    """
    # Convert magnitude and phase into a complex number: mag * e^(j * 2π * phase)
    complex_coeffs = mag * np.exp(1j * 2 * np.pi * phase)
    
    # Calculate energy as the sum of the squared magnitude of the complex coefficients
    energy = np.sum(np.abs(complex_coeffs) ** 2)
    return energy


def calculate_wer(current_energy, previous_energy):
    """
    Calculate the Waveform Energy Ratio (WER) between two consecutive levels.

    Args:
    current_energy (float): The energy of the current level.
    previous_energy (float): The energy of the previous level.

    Returns:
    float: The ratio of the current energy to the previous energy.
    """
    if previous_energy > 0:
        return current_energy / previous_energy
    else:
        return 0  # Return 0 if previous energy is 0 to avoid division by zero


def calculate_energy_based_features(coeffs, num_levels):
    """
    Calculate energy-based features for wavelet coefficients (magnitude and phase) across multiple levels.

    Args:
    coeffs (dict): Dictionary containing magnitude and phase wavelet coefficients.
    num_levels (int): Number of wavelet decomposition levels.

    Returns:
    dict: Dictionary with energy-based features and WER for each level, structured by coefficient type (cA, cH, cV, cD).
    """

    energy_level_values = []  # To store energy for each level and coefficient type
    total_energy = 0  # To calculate the total energy across all levels and coefficient types

    # Dictionary to store all energy-based features for each level and coefficient type
    energy_features = {}
    
    coeffs_mag, coeffs_phase = coeffs
    cA_mag, cA_phase = coeffs_mag[0], coeffs_phase[0]
    energy_cA = calculate_energy(cA_mag, cA_phase)
    total_energy += energy_cA

    # Store energy for cA (approximation coefficients)
    energy_features[f'level_{num_levels}'] = {'cA_energy': energy_cA}

    energy_level_values.append(energy_cA)  # Add energy to the list for WER calculation

    # Process the detail coefficients (cH, cV, cD) for levels 1 to num_levels
    for level in range(1, num_levels+1):
        # TODO check if this is ok (extraction of the phase and mag):
        # Get the tuple of detail coefficients for this level (cH, cV, cD)
        cH_mag, cV_mag, cD_mag = coeffs_mag[-level]
        cH_phase, cV_phase, cD_phase = coeffs_phase[-level]

        # (cH_mag, cH_phase), (cV_mag, cV_phase), (cD_mag, cD_phase) = (coeffs_mag[-level], coeffs_phase[-level])
        
        energy_cH = calculate_energy(cH_mag, cH_phase)
        energy_cV = calculate_energy(cV_mag, cV_phase)
        energy_cD = calculate_energy(cD_mag, cD_phase)

        sum_energies = energy_cD + energy_cV +energy_cH

        total_energy += sum_energies

        # Store energy values for each type of coefficient (cH, cV, cD)
        energy_features[f'level_{level}'] = {
            'cH_energy': energy_cH,
            'cV_energy': energy_cV,
            'cD_energy': energy_cD,
            'total_energy': sum_energies
        }
        
        # Add the total energy for this level to the list (for WER calculation)
        energy_level_values.append(sum_energies)

    for level in range(1, num_levels+1):
        current_energy = energy_level_values[level]
        previous_energy = energy_level_values[level]
        wer = calculate_wer(current_energy, previous_energy)
        energy_features[f'level_{level}']['wer'] = wer

    return energy_features



# ============================================================
#                   Statistical Based Features
# ============================================================


def calculate_basic_statistical_features(coeffs, num_levels):
    statistical_features = {}

    coeffs_mag, coeffs_phase = coeffs
    cA_mag, cA_phase = coeffs_mag[0], coeffs_phase[0]
    cA_complex = make_complex(cA_mag, cA_phase)

    # Utility function to make values serializable
    def make_serializable(value):
        if isinstance(value, (complex, np.complex128)):
            # Convert complex numbers into a dictionary with real and imaginary parts
            return {'real': make_serializable(value.real), 'imag': make_serializable(value.imag)}
        elif isinstance(value, np.ndarray):
            # Convert numpy arrays to lists
            return [make_serializable(item) for item in value.tolist()]
        elif isinstance(value, (np.float64, np.int64)):
            # Convert numpy specific data types to native Python types
            return value.item()
        elif isinstance(value, list):
            # Recursively process each item in the list
            return [make_serializable(item) for item in value]
        elif isinstance(value, dict):
            return {key : make_serializable(val) for key, val in value.items()}
        else:
            # Return other types as is
            return value
    
    # Compute cA features
    statistical_features[f'level_{num_levels}'] = {
        'cA_max': make_serializable(np.max(cA_mag) * np.exp(1j * 2 * np.pi * np.max(cA_phase))),
        'cA_min': make_serializable(np.min(cA_mag) * np.exp(1j * 2 * np.pi * np.max(cA_phase))),
        'cA_mean': make_serializable(np.mean(cA_complex)),
        'cA_variance': make_serializable(np.var(cA_complex)),
        'cA_std': make_serializable(np.std(cA_complex)),
        'cA_range': make_serializable(np.max(cA_mag) * np.exp(1j * 2 * np.pi * np.max(cA_phase)) - np.min(cA_mag) * np.exp(1j * 2 * np.pi * np.min(cA_phase))),
        'cA_median': make_serializable(np.median(cA_complex)),
        'cA_skewness': make_serializable(stats.skew(np.real(cA_complex)) + 1j * stats.skew(np.imag(cA_complex))),
        'cA_kurtosis': make_serializable(stats.kurtosis(np.real(cA_complex)) + 1j * stats.kurtosis(np.imag(cA_complex)))
    }

    # Process detail coefficients for each level
    for level in range(1, num_levels + 1):
        cH_mag, cV_mag, cD_mag = coeffs_mag[-level]
        cH_phase, cV_phase, cD_phase = coeffs_phase[-level]

        # Initialize the dictionary for the current level if not already present
        if f'level_{level}' not in statistical_features:
            statistical_features[f'level_{level}'] = {}

        # Process cH
        cH_complex = make_complex(cH_mag, cH_phase)
        statistical_features[f'level_{level}'].update({
            'cH_max': make_serializable(np.max(cH_mag) * np.exp(1j * 2 * np.pi * np.max(cH_phase))),
            'cH_min': make_serializable(np.min(cH_mag) * np.exp(1j * 2 * np.pi * np.max(cH_phase))),
            'cH_mean': make_serializable(np.mean(cH_complex)),
            'cH_variance': make_serializable(np.var(cH_complex)),
            'cH_std': make_serializable(np.std(cH_complex)),
            'cH_range': make_serializable(np.max(cH_mag) * np.exp(1j * 2 * np.pi * np.max(cH_phase)) - np.min(cH_mag) * np.exp(1j * 2 * np.pi * np.max(cH_phase))),
            'cH_median': make_serializable(np.median(cH_complex)),
            'cH_skewness': make_serializable(stats.skew(np.real(cH_complex)) + 1j * stats.skew(np.imag(cH_complex))),
            'cH_kurtosis': make_serializable(stats.kurtosis(np.real(cH_complex)) + 1j * stats.kurtosis(np.imag(cH_complex)))
        })

        # Process cV
        cV_complex = make_complex(cV_mag, cV_phase)
        statistical_features[f'level_{level}'].update({
            'cV_max': make_serializable(np.max(cV_mag) * np.exp(1j * 2 * np.pi * np.max(cV_phase))),
            'cV_min': make_serializable(np.min(cV_mag) * np.exp(1j * 2 * np.pi * np.max(cV_phase))),
            'cV_mean': make_serializable(np.mean(cV_complex)),
            'cV_variance': make_serializable(np.var(cV_complex)),
            'cV_std': make_serializable(np.std(cV_complex)),
            'cV_range': make_serializable(np.max(cV_mag) * np.exp(1j * 2 * np.pi * np.max(cV_phase)) - np.min(cV_mag) * np.exp(1j * 2 * np.pi * np.max(cV_phase))),
            'cV_median': make_serializable(np.median(cV_complex)),
            'cV_skewness': make_serializable(stats.skew(np.real(cV_complex)) + 1j * stats.skew(np.imag(cV_complex))),
            'cV_kurtosis': make_serializable(stats.kurtosis(np.real(cV_complex)) + 1j * stats.kurtosis(np.imag(cV_complex)))
        })

        # Process cD
        cD_complex = make_complex(cD_mag, cD_phase)
        statistical_features[f'level_{level}'].update({
            'cD_max': make_serializable(np.max(cD_mag) * np.exp(1j * 2 * np.pi * np.max(cD_phase))),
            'cD_min': make_serializable(np.min(cD_mag) * np.exp(1j * 2 * np.pi * np.max(cD_phase))),
            'cD_mean': make_serializable(np.mean(cD_complex)),
            'cD_variance': make_serializable(np.var(cD_complex)),
            'cD_std': make_serializable(np.std(cD_complex)),
            'cD_range': make_serializable(np.max(cD_mag) * np.exp(1j * 2 * np.pi * np.max(cD_phase)) - np.min(cD_mag) * np.exp(1j * 2 * np.pi * np.max(cD_phase))),
            'cD_median': make_serializable(np.median(cD_complex)),
            'cD_skewness': make_serializable(stats.skew(np.real(cD_complex)) + 1j * stats.skew(np.imag(cD_complex))),
            'cD_kurtosis': make_serializable(stats.kurtosis(np.real(cD_complex)) + 1j * stats.kurtosis(np.imag(cD_complex)))
        })

    return statistical_features




# def calculate_basic_statistical_features(coeffs, num_levels):
#     """
#     Calculate basic statistical features for wavelet coefficients from a NumPy file.

#     Args:
#     numpy_file (str): Path to the NumPy file containing the wavelet coefficients.
#     level (int, optional): The level of wavelet transform. Default is 2.

#     Returns:
#     list: A list containing the basic statistical features.
#     """
#     # Dictionary to store all statistical features for each level and coefficient type
#     statistical_features = {}

#     coeffs_mag, coeffs_phase = coeffs
#     cA_mag, cA_phase = coeffs_mag[0], coeffs_phase[0]
#     cA_complex = make_complex(cA_mag, cA_phase)
#     mean_cA = np.mean(cA_complex)
#     var_cA = np.var(cA_complex)
#     std_cA = np.std(cA_complex)
#     max_cA = np.max(cA_mag) * np.exp(1j * 2 * np.pi * np.max(cA_phase))
#     min_cA = np.min(cA_mag) * np.exp(1j * 2 * np.pi * np.min(cA_phase))
#     range_cA = max_cA - min_cA
#     median_cA = np.median(cA_complex)
#     skew_cA = stats.skew(np.real(cA_complex)) + 1j * stats.skew(np.imag(cA_complex)) # Skewness of the real part
#     kurtosis_cA = stats.kurtosis(np.real(cA_complex)) + 1j * stats.kurtosis(np.imag(cA_complex))# Kurtosis of the real part

#     statistical_features[f'level_{num_levels}'] = {
#         'cA_mean': mean_cA,
#         'cA_variance': var_cA,
#         'cA_std': std_cA, 
#         'cA_range': range_cA,
#         'cA_median': median_cA,
#         'cA_skewness': skew_cA,
#         'cA_kurtosis': kurtosis_cA
#     }

#     for level in range(1, num_levels+1):
#         # Initialize the dictionary for the current level if not already present
#         if f'level_{level}' not in statistical_features:
#             statistical_features[f'level_{level}'] = {}
#         # Get the tuple of detail coefficients for this level (cH, cV, cD)
#         cH_mag, cV_mag, cD_mag = coeffs_mag[-level]
#         cH_phase, cV_phase, cD_phase = coeffs_phase[-level]
#         #(cH_mag, cH_phase), (cV_mag, cV_phase), (cD_mag, cD_phase) = (coeffs_mag[-level], coeffs_phase[-level])

#         # Process cH:
#         cH_complex = make_complex(cH_mag, cH_phase)

#         mean_cH = np.mean(cH_complex)
#         var_cH = np.var(cH_complex)
#         std_cH = np.std(cH_complex)
#         max_cH = np.max(cH_mag) * np.exp(1j * 2 * np.pi * np.max(cH_phase))
#         min_cH = np.min(cH_mag) * np.exp(1j * 2 * np.pi * np.max(cH_phase))
#         range_cH = max_cH - min_cH,
#         median_cH = np.median(cH_complex)
#         skew_cH = stats.skew(np.real(cH_complex)) + 1j * stats.skew(np.imag(cH_complex)) # Skewness of the real part
#         kurtosis_cH = stats.kurtosis(np.real(cH_complex)) + 1j * stats.skew(np.imag(cH_complex))# Kurtosis of the real part

#         statistical_features[f'level_{level}'].update({
#             'cH_max': max_cH,
#             'cH_min': min_cH,
#             'cH_mean': mean_cH,
#             'cH_variance': var_cH,
#             'cH_std': std_cH, 
#             'cH_range': range_cH,
#             'cH_median': median_cH,
#             'cH_skewness': skew_cH,
#             'cH_kurtosis': kurtosis_cH
#         })

#         # Process cV:
#         cV_complex = make_complex(cV_mag, cV_phase)

#         mean_cV = np.mean(cV_complex)
#         var_cV = np.var(cV_complex)
#         std_cV = np.std(cV_complex)
#         max_cV = np.max(cV_mag) * np.exp(1j * 2 * np.pi * np.max(cV_phase))
#         min_cV = np.min(cV_mag) * np.exp(1j * 2 * np.pi * np.max(cV_phase))
#         range_cV = max_cV - min_cV
#         median_cV = np.median(cV_complex)
#         skew_cV = stats.skew(np.real(cV_complex)) + 1j * stats.skew(np.imag(cV_complex)) # Skewness of the real part
#         kurtosis_cV = stats.kurtosis(np.real(cV_complex)) + 1j * stats.skew(np.imag(cV_complex)) # Kurtosis of the real part

#         statistical_features[f'level_{level}'].update({
#             'cV_max': max_cV,
#             'cV_min': min_cV,
#             'cV_mean': mean_cV,
#             'cV_variance': var_cV,
#             'cV_std': std_cV, 
#             'cV_range': range_cV,
#             'cV_median': median_cV,
#             'cV_skewness': skew_cV,
#             'cV_kurtosis': kurtosis_cV
#         })

#         # Process cD:
#         cD_complex = make_complex(cD_mag, cD_phase)

#         mean_cD = np.mean(cD_complex)
#         var_cD = np.var(cD_complex)
#         std_cD = np.std(cD_complex)
#         max_cD = np.max(cD_mag) * np.exp(1j * 2 * np.pi * np.max(cD_phase))
#         min_cD = np.min(cD_mag) * np.exp(1j * 2 * np.pi * np.max(cD_phase))
#         range_cD = max_cD - min_cD
#         median_cD = np.median(cD_complex)
#         skew_cD = stats.skew(np.real(cD_complex)) + 1j * stats.skew(np.imag(cD_complex))# Skewness of the real part
#         kurtosis_cD = stats.kurtosis(np.real(cD_complex)) + 1j * stats.skew(np.imag(cD_complex))# Kurtosis of the real part

#         statistical_features[f'level_{level}'].update({
#             'cD_max': max_cD,
#             'cD_min': min_cD,
#             'cD_mean': mean_cD,
#             'cD_variance': var_cD,
#             'cD_std': std_cD, 
#             'cD_range': range_cD,
#             'cD_median': median_cD,
#             'cD_skewness': skew_cD,
#             'cD_kurtosis': kurtosis_cD
#         })

#     return statistical_features

# ============================================================
#               Advanced Statistical Based Features
# ============================================================

def calculate_iqr(coeffs):
    """
    Calculate the Interquartile Range (IQR) for the given coefficients.
    """
    Q1 = np.percentile(coeffs, 25)
    Q3 = np.percentile(coeffs, 75)
    return Q3 - Q1


def calculate_coefficient_of_variation(coeffs):
    """
    Calculate the Coefficient of Variation for the given coefficients.
    """
    mean = np.mean(coeffs)
    std = np.std(coeffs)
    return std / mean if mean != 0 else 0


def calculate_approximate_entropy(coeffs, m=2, r=0.2):
    """
    Calculate the Approximate Entropy for the given coefficients.
    """
    N = len(coeffs)
    
    def _phi(m):
        x = np.array([coeffs[i:i + m] for i in range(N - m + 1)])
        C = np.sum(np.max(np.abs(x[:, None] - x[None, :]), axis=2) <= r, axis=0) / (N - m + 1)
        return np.sum(np.log(C)) / (N - m + 1)

    return abs(_phi(m) - _phi(m + 1))


def calculate_hurst_exponent(coeffs):
    """
    Calculate the Hurst Exponent using pyeeg, if available.
    """
    try:
        import pyeeg
        return pyeeg.hurst(coeffs)
    except ImportError:
        return None


def calculate_skewness_kurtosis(coeffs):
    """
    Calculate the Skewness and Kurtosis for the given coefficients.
    """
    skewness = stats.skew(coeffs)
    kurtosis = stats.kurtosis(coeffs)
    return skewness, kurtosis


def aggrigate_advanced_statistical_measures(coeffs_mag, coeffs_phase, coeff_name):
    """
    Extracts all statistical features (IQR, Skewness, Kurtosis, etc.) for the given coefficients.
    
    Returns a dictionary of features for both magnitude and phase.
    """
    features = {}

    # Magnitude-based features
    features[f'{coeff_name}_mag_iqr'] = calculate_iqr(coeffs_mag)
    features[f'{coeff_name}_mag_coefficient_of_variation'] = calculate_coefficient_of_variation(coeffs_mag)
    features[f'{coeff_name}_mag_approximate_entropy'] = calculate_approximate_entropy(coeffs_mag)

    skewness_mag, kurtosis_mag = calculate_skewness_kurtosis(coeffs_mag)
    features[f'{coeff_name}_mag_skewness'] = skewness_mag
    features[f'{coeff_name}_mag_kurtosis'] = kurtosis_mag
    features[f'{coeff_name}_mag_hurst_exponent'] = calculate_hurst_exponent(coeffs_mag)

    # Phase-based features (optional)
    features[f'{coeff_name}_phase_iqr'] = calculate_iqr(coeffs_phase)
    skewness_phase, kurtosis_phase = calculate_skewness_kurtosis(coeffs_phase)
    features[f'{coeff_name}_phase_skewness'] = skewness_phase
    features[f'{coeff_name}_phase_kurtosis'] = kurtosis_phase
    features[f'{coeff_name}_phase_approximate_entropy'] = calculate_approximate_entropy(coeffs_phase)

    return features


def calculate_advanced_statistical_features(coeffs, level_num):
    """
    Calculate advanced statistical features for wavelet coefficients (including cA, cH, cV, cD) across all levels.
    
    Args:
    coeffs (list): List of wavelet coefficients, where:
                   - `coeffs[0]`: Approximation coefficients (cA) at level 0
                   - `coeffs[1:]`: Tuples of (cH, cV, cD) for each subsequent level.
    num_levels (int): Number of wavelet decomposition levels.

    Returns:
    dict: Dictionary with advanced statistical features for each level and coefficient type.
    """
    # Process the approximation coefficients (cA) at the nth level (only present in the first element of coeffs)
    
    statistical_features = {}

    mag, phase = coeffs

    cA_mag, cA_phase = mag[0], phase[0]

    statistical_features['cA_level_0'] = aggrigate_advanced_statistical_measures(cA_mag, cA_phase, 'cA')

    for level in range(1, level_num+1):
        (cH_mag, cV_mag, cD_mag) = mag[-level]
        (cH_phase, cV_phase, cD_phase) = phase[-level]

        # Extract features for each type of detail coefficient
        statistical_features[f'cH_level_{level}'] = aggrigate_advanced_statistical_measures(cH_mag, cH_phase, 'cH')
        statistical_features[f'cV_level_{level}'] = aggrigate_advanced_statistical_measures(cV_mag, cV_phase, 'cV')
        statistical_features[f'cD_level_{level}'] = aggrigate_advanced_statistical_measures(cD_mag, cD_phase, 'cD')

    return statistical_features

# ============================================================
#                      Temporal Features
# ============================================================

def calculate_max_min_locations(mag):
    """
    Calculate the time locations of max and min values of the wavelet coefficients.
    
    Args:
    mag (np.array): Magnitude coefficients.

    Returns:
    tuple: Max location, Min location.
    """
    # Find the max and min locations
    max_location = np.argmax(mag)
    min_location = np.argmin(mag)
    
    return max_location, min_location


def calculate_zero_crossing_rate(complex_coeffs):
    """
    Calculate the zero crossing rate of the wavelet coefficients.
    
    Args:
    mag (np.array): Magnitude coefficients.
    phase (np.array): Phase coefficients.

    Returns:
    float: Zero crossing rate.
    """
    # Calculate the zero crossing rate
    zero_crossings = np.sum(np.diff(np.sign(np.real(complex_coeffs))) != 0) / len(complex_coeffs)
    
    return zero_crossings


def calculate_waveform_length(complex_coeffs):
    """
    Calculate the waveform length from magnitude and phase.
    
    Args:
    mag (np.array): Magnitude coefficients.
    phase (np.array): Phase coefficients.

    Returns:
    float: Waveform length.
    """
    # Calculate waveform length (sum of absolute differences between consecutive points)
    waveform_length = np.sum(np.abs(np.diff(complex_coeffs)))
    
    return waveform_length


def calculate_temporal_features(coeffs, num_levels):
    """
    Calculate temporal features for wavelet coefficients (magnitude and phase) across multiple levels.

    Args:
    coeffs (list): List of wavelet coefficients where the first element is the approximation coefficients (cA)
                   and the rest are tuples of detail coefficients (cH, cV, cD) for each level.
                   Each element is a tuple containing (magnitude, phase).
    num_levels (int): Number of wavelet decomposition levels.

    Returns:
    dict: Dictionary with temporal features for each level, structured by coefficient type (cA, cH, cV, cD).
    """
    # Dictionary to store temporal features for each level and coefficient type
    temporal_features = {}

    coeffs_mag, coeffs_phase = coeffs
    # Process the approximation coefficients (cA) at the nth level (only present in the first element of coeffs)
    cA_mag, cA_phase = coeffs_mag[0], coeffs_phase[0]  # Approximation coefficients

    cA_complex = make_complex(cA_mag, cA_phase)
    # Calculate temporal features for cA (approximation coefficients)
    waveform_length = calculate_waveform_length(cA_complex)
    zero_crossings = calculate_zero_crossing_rate(cA_complex)
    max_location, min_location = calculate_max_min_locations(cA_mag)

    temporal_features[f'level_{num_levels}'] = {
        'cA_waveform_length': waveform_length,
        'cA_zero_crossings': zero_crossings,
        'cA_max_location': max_location,
        'cA_min_location': min_location
    }

    for level in range (1, num_levels+1):
        # Get the tuple of detail coefficients for this level (cH, cV, cD)
        (cH_mag, cH_phase), (cV_mag, cV_phase), (cD_mag, cD_phase) = (coeffs_mag[-level], coeffs_phase[-level])

        cH_complex = make_complex(cH_mag, cH_phase)
        cV_complex = make_complex(cV_mag, cV_phase)
        cD_complex = make_complex(cD_mag, cD_phase)

        # Calculate temporal features for each coefficient type (cH, cV, cD)
        waveform_length_H = calculate_waveform_length(cH_complex)
        zero_crossings_H = calculate_zero_crossing_rate(cH_complex)
        max_location_H, min_location_H = calculate_max_min_locations(cH_mag)

        waveform_length_V = calculate_waveform_length(cV_complex)
        zero_crossings_V = calculate_zero_crossing_rate(cV_complex)
        max_location_V, min_location_V = calculate_max_min_locations(cV_mag)

        waveform_length_D = calculate_waveform_length(cD_complex)
        zero_crossings_D = calculate_zero_crossing_rate(cD_complex)
        max_location_D, min_location_D = calculate_max_min_locations(cD_mag)

        # Store temporal features for each coefficient type (cH, cV, cD)
        temporal_features[f'level_{level}'] = {
            'cH_waveform_length': waveform_length_H,
            'cH_zero_crossings': zero_crossings_H,
            'cH_max_location': max_location_H,
            'cH_min_location': min_location_H,

            'cV_waveform_length': waveform_length_V,
            'cV_zero_crossings': zero_crossings_V,
            'cV_max_location': max_location_V,
            'cV_min_location': min_location_V,

            'cD_waveform_length': waveform_length_D,
            'cD_zero_crossings': zero_crossings_D,
            'cD_max_location': max_location_D,
            'cD_min_location': min_location_D
        }

    return temporal_features


# ============================================================
#                      Frequency Features
# ============================================================

def instantaneous_frequency(complex_coeffs):
    """
    Calculate the instantaneous frequency from wavelet coefficients.
    
    Args:
    wavelet_coeffs (numpy array): Array of wavelet coefficients for a given level.
    
    Returns:
    numpy array: Instantaneous frequency over time.
    """
    # Calculate the analytic signal using Hilbert transform
    analytic_signal = hilbert(complex_coeffs)
    
    # Extract the phase of the analytic signal
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    
    # Compute the derivative of the phase to get the instantaneous frequency
    instantaneous_freq = np.diff(instantaneous_phase) / (2.0 * np.pi)
    
    return instantaneous_freq


def wavelet_spectrum(fft_coeffs):
    """
    Calculate the wavelet spectrum (spectral content) for wavelet coefficients.
    
    Args:
    wavelet_coeffs (numpy array): Array of wavelet coefficients for a given level.
    
    Returns:
    numpy array: Energy spectrum of the wavelet coefficients across frequency bands.
    """
    # Apply the Fourier Transform to the wavelet coefficients
    spectrum = fft_coeffs ** 2
    
    # Return the energy spectrum (sum of squared magnitudes of the Fourier coefficients)
    return spectrum


def dominant_frequency_and_ratios(fft_coeffs, fft_freq_bands):
    # Find the dominant frequency (the frequency with the highest magnitude)
    dominant_freq = fft_freq_bands[np.argmax(fft_coeffs)]

    # Calculate energy in different frequency bands
    low_freq_energy = np.sum(fft_coeffs[fft_freq_bands < 0.3] ** 2)
    mid_freq_energy = np.sum(fft_coeffs[(fft_freq_bands >= 0.3) & (fft_freq_bands < 0.7)] ** 2)
    high_freq_energy = np.sum(fft_coeffs[fft_freq_bands >= 0.7] ** 2)
    
    # Calculate low/mid and low/high ratios
    if low_freq_energy > 0:
        low_mid_ratio = mid_freq_energy / low_freq_energy
        low_high_ratio = high_freq_energy / low_freq_energy
    else:
        low_mid_ratio = 0
        low_high_ratio = 0

    return dominant_freq, low_mid_ratio, low_high_ratio


def calculate_frequency_features(coeffs, num_levels):
    # Dictionary to store frequency-based features for each level
    frequency_features = {}

    coeffs_mag, coeffs_phase = coeffs

    # Process the approximation coefficients (cA) at the nth level (only present in the first element of coeffs)
    cA_mag, cA_phase = coeffs_mag[0], coeffs_phase[0]  # Approximation coefficients

    # Combine mag and phase into a complex signal
    cA_complex = make_complex(cA_mag, cA_phase)

    # Apply the Fourier Transform to the complex coefficients
    fft_coeffs_cA = calc_abs_fourier(cA_complex)

    # Get the frequency bands
    freq_bands_cA = fftfreq(len(cA_complex))

    # Calculate frequency-based features for cA (approximation coefficients)
    dom_freq, low_mid_ratio, low_high_ratio = dominant_frequency_and_ratios(fft_coeffs_cA, freq_bands_cA)
    freq_inst = instantaneous_frequency(cA_complex)
    spectrum = wavelet_spectrum(fft_coeffs_cA)

    frequency_features[f'level_{num_levels}'] = {
        'cA_dominant_frequency': dom_freq,
        'cA_low_mid_ratio': low_mid_ratio,
        'cA_low_high_ratio': low_high_ratio,
        'cA_instantaneous_frequency': freq_inst,
        'cA_wavelet_spectrum': spectrum
    }

    for level in range(1, num_levels+1):
        # Get the tuple of detail coefficients for this level (cH, cV, cD)
        (cH_mag, cH_phase), (cV_mag, cV_phase), (cD_mag, cD_phase) = (coeffs_mag[-level], coeffs_phase[-level])
        
        # Combine mag and phase into a complex signal
        cH_complex = make_complex(cH_mag, cH_phase)
        cV_complex = make_complex(cV_mag, cV_phase)
        cD_complex = make_complex(cD_mag, cD_phase)

        # Apply the Fourier Transform to the complex coefficients
        cH_fft_coeffs = calc_abs_fourier(cH_complex)
        cV_fft_coeffs = calc_abs_fourier(cV_complex)
        cD_fft_coeffs = calc_abs_fourier(cD_complex)

        # Get the frequency bands
        cH_freq_bands = fftfreq(len(cH_complex))
        cV_freq_bands = fftfreq(len(cV_complex))
        cD_freq_bands = fftfreq(len(cD_complex))

        # Calculate frequency-based features for each coefficient type (cH, cV, cD)
        dom_freq_H, low_mid_ratio_H, low_high_ratio_H = dominant_frequency_and_ratios(cH_fft_coeffs, cH_freq_bands)
        dom_freq_V, low_mid_ratio_V, low_high_ratio_V = dominant_frequency_and_ratios(cV_fft_coeffs, cV_freq_bands)
        dom_freq_D, low_mid_ratio_D, low_high_ratio_D = dominant_frequency_and_ratios(cD_fft_coeffs, cD_freq_bands)

        freq_inst_H = instantaneous_frequency(cH_complex)
        freq_inst_V = instantaneous_frequency(cV_complex)
        freq_inst_D = instantaneous_frequency(cD_complex)

        spectrum_H = wavelet_spectrum(cH_fft_coeffs)
        spectrum_V = wavelet_spectrum(cV_fft_coeffs)
        spectrum_D = wavelet_spectrum(cD_fft_coeffs)

        # Store frequency features for each coefficient type (cH, cV, cD)
        frequency_features[f'level_{level}'] = {
            'cH_dominant_frequency': dom_freq_H,
            'cH_low_mid_ratio': low_mid_ratio_H,
            'cH_low_high_ratio': low_high_ratio_H,
            'cH_instantaneous_frequency': freq_inst_H,
            'cH_wavelet_spectrum': spectrum_H,

            'cV_dominant_frequency': dom_freq_V,
            'cV_low_mid_ratio': low_mid_ratio_V,
            'cV_low_high_ratio': low_high_ratio_V,
            'cV_instantaneous_frequency': freq_inst_V,
            'cV_wavelet_spectrum': spectrum_V,

            'cD_dominant_frequency': dom_freq_D,
            'cD_low_mid_ratio': low_mid_ratio_D,
            'cD_low_high_ratio': low_high_ratio_D,
            'cD_instantaneous_frequency': freq_inst_D,
            'cD_wavelet_spectrum': spectrum_D
        }

    return frequency_features


# ============================================================
#               Information Theoretic Features
# ============================================================

def shannon_entropy(prob_dist):
    """
    Calculate the Shannon entropy of the wavelet coefficients.
    
    Args:
    wavelet_coeffs (numpy array): Array of wavelet coefficients.
    
    Returns:
    float: Shannon entropy of the coefficients.
    """
    # Normalize the wavelet coefficients to get a probability distribution
    # prob_dist, _ = np.histogram(wavelet_coeffs, bins=100, density=True)
    
    # Remove zeros from the probability distribution (as log(0) is undefined)
    prob_dist = prob_dist[prob_dist > 0]
    
    # Shannon entropy formula: -sum(p * log(p))
    entropy = -np.sum(prob_dist * np.log(prob_dist))
    
    return entropy


def kl_divergence(p_hist, bin_edges, coeff_mean, coeff_std , reference='gaussian'):
    """
    Calculate the Kullback-Leibler (KL) divergence between the wavelet coefficient 
    distribution and a reference distribution.
    
    Args:
    wavelet_coeffs (numpy array): Array of wavelet coefficients.
    reference (str): Reference distribution. Options: 'gaussian' (default), 'uniform', or custom.
    
    Returns:
    float: KL divergence value.
    """
    p_hist = p_hist + 1e-10  # Add a small constant to avoid log(0)
    
    # Compute the bin centers (for calculating q distribution)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Reference distribution: Gaussian (normal distribution)
    if reference == 'gaussian':
        # mu, sigma = coeff_mean, coeff_std
        q_dist = norm.pdf(bin_centers, coeff_mean, coeff_std)
    # elif reference == 'uniform':
    #     q_dist = np.ones_like(bin_centers) / len(bin_centers)  # Uniform distribution
    else:
        raise ValueError("Unsupported reference distribution.")
    
    # Normalize q distribution
    q_dist = q_dist / np.sum(q_dist)
    
    # KL divergence formula: sum(p * log(p / q))
    kl_div = np.sum(p_hist * np.log(p_hist / q_dist))
    
    return kl_div


def mutual_information(wavelet_coeffs_x, wavelet_coeffs_y):
    """
    Calculate the mutual information between two sets of wavelet coefficients.
    
    Args:
    wavelet_coeffs_x (numpy array): Array of wavelet coefficients for level X.
    wavelet_coeffs_y (numpy array): Array of wavelet coefficients for level Y.
    
    Returns:
    float: Mutual information between the two levels of coefficients.
    """
    # Discretize the coefficients into bins (for computing mutual information)
    bins = 100
    x_binned = np.digitize(wavelet_coeffs_x, np.histogram_bin_edges(wavelet_coeffs_x, bins=bins))
    y_binned = np.digitize(wavelet_coeffs_y, np.histogram_bin_edges(wavelet_coeffs_y, bins=bins))
    
    # Mutual information score (normalized)
    mi = mutual_info_score(x_binned, y_binned)
    
    return mi


def conditional_entropy(wavelet_coeffs_t, wavelet_coeffs_t_minus_1):
    """
    Calculate the conditional entropy between two levels of wavelet coefficients.
    
    Args:
    wavelet_coeffs_t (numpy array): Array of wavelet coefficients for the current level.
    wavelet_coeffs_t_minus_1 (numpy array): Array of wavelet coefficients for the previous level.
    
    Returns:
    float: Conditional entropy between the current and previous levels.
    """
    # Discretize the coefficients into bins (for entropy calculation)
    bins = 100
    t_binned = np.digitize(wavelet_coeffs_t, np.histogram_bin_edges(wavelet_coeffs_t, bins=bins))
    t_minus_1_binned = np.digitize(wavelet_coeffs_t_minus_1, np.histogram_bin_edges(wavelet_coeffs_t_minus_1, bins=bins))
    
    # Calculate joint entropy (H(C_t, C_{t-1}))
    joint_hist, _, _ = np.histogram2d(t_binned, t_minus_1_binned, bins=bins)
    joint_entropy = -np.sum(joint_hist.ravel() * np.log2(joint_hist.ravel() + 1e-10))
    
    # Calculate marginal entropy of the previous level (H(C_{t-1}))
    t_minus_1_hist, _ = np.histogram(t_minus_1_binned, bins=bins)
    t_minus_1_entropy = -np.sum(t_minus_1_hist * np.log2(t_minus_1_hist + 1e-10))

    # Conditional entropy formula: H(C_t | C_{t-1}) = H(C_t, C_{t-1}) - H(C_{t-1})
    conditional_entropy_value = joint_entropy - t_minus_1_entropy
    
    return conditional_entropy_value


def calculate_information_theoretic_features(coeffs, num_levels):
    """
    Calculate information_theoretic-based features for wavelet coefficients (magnitude and phase) across levels.
    
    Args:
    coeffs (list): List of tuples where each element is (magnitude_coeffs, phase_coeffs) for each level.
    num_levels (int): Number of wavelet levels.

    Returns:
    dict: Dictionary with information_theoretic-based features for each level.
    """
    # Dictionary to store information-theoretic features for each level
    information_theoretic = {}
    
    prev_mag_wavelet = None
    prev_phase_wavelet = None

    mag, phase = coeffs

    cA_mag, cA_phase = mag[0], phase[0]

    # Compute the probability distributions for magnitude and phase coefficients
    p_cA_hist_mag, bin_cA_edges_mag = np.histogram(cA_mag, bins=100, density=True)
    p_cA_hist_phase, bin_cA_edges_phase = np.histogram(cA_phase, bins=100, density=True)

    # Mean and standard deviation for KL divergence (for magnitude and phase)
    cA_mag_mean, cA_mag_std = np.mean(cA_mag), np.std(cA_mag)
    cA_phase_mean, cA_phase_std = np.mean(cA_phase), np.std(cA_phase)

    # KL Divergence
    cA_kl_mag = kl_divergence(p_cA_hist_mag, bin_cA_edges_mag, cA_mag_mean, cA_mag_std, reference='gaussian')
    cA_kl_phase = kl_divergence(p_cA_hist_phase, bin_cA_edges_phase, cA_phase_mean, cA_phase_std, reference='gaussian')

    # Shannon Entropy
    cA_shannon_mag = shannon_entropy(p_cA_hist_mag)
    cA_shannon_phase = shannon_entropy(p_cA_hist_phase)

    information_theoretic[f'level_{num_levels}'] = {
        'cA_KL_magnitude': cA_kl_mag,
        'cA_KL_phase': cA_kl_phase,
        'cA_shannon_magnitude': cA_shannon_mag,
        'cA_shannon_phase': cA_shannon_phase
    }

    # Mutual Information (between levels, both magnitude and phase)
    mi_mag = 0
    mi_phase = 0
        
    # Conditional Entropy (for magnitude and phase)
    ce_mag = 0
    ce_phase = 0
    
    for level in range(1, num_levels+1):
        # Get the tuple of detail coefficients for this level (cH, cV, cD)
        (cH_mag, cH_phase), (cV_mag, cV_phase), (cD_mag, cD_phase) = (mag[-level], phase[-level])

        # cH:
        # Compute the probability distributions for magnitude and phase coefficients
        p_cH_hist_mag, bin_cH_edges_mag = np.histogram(cH_mag, bins=100, density=True)
        p_cH_hist_phase, bin_cH_edges_phase = np.histogram(cH_phase, bins=100, density=True)

        # Mean and standard deviation for KL divergence (for magnitude and phase)
        cH_mag_mean, cH_mag_std = np.mean(cH_mag), np.std(cH_mag)
        cH_phase_mean, cH_phase_std = np.mean(cH_phase), np.std(cH_phase)

        # KL Divergence
        cH_kl_mag = kl_divergence(p_cH_hist_mag, bin_cH_edges_mag, cH_mag_mean, cH_mag_std, reference='gaussian')
        cH_kl_phase = kl_divergence(p_cH_hist_phase, bin_cH_edges_phase, cH_phase_mean, cH_phase_std, reference='gaussian')

        # Shannon Entropy
        cH_shannon_mag = shannon_entropy(p_cH_hist_mag)
        cH_shannon_phase = shannon_entropy(p_cH_hist_phase)

        # cV:
        # Compute the probability distributions for magnitude and phase coefficients
        p_cV_hist_mag, bin_cV_edges_mag = np.histogram(cV_mag, bins=100, density=True)
        p_cV_hist_phase, bin_cV_edges_phase = np.histogram(cV_phase, bins=100, density=True)

        # Mean and standard deviation for KL divergence (for magnitude and phase)
        cV_mag_mean, cV_mag_std = np.mean(cV_mag), np.std(cV_mag)
        cV_phase_mean, cV_phase_std = np.mean(cV_phase), np.std(cV_phase)

        # KL Divergence
        cV_kl_mag = kl_divergence(p_cV_hist_mag, bin_cV_edges_mag, cV_mag_mean, cV_mag_std, reference='gaussian')
        cV_kl_phase = kl_divergence(p_cV_hist_phase, bin_cV_edges_phase, cV_phase_mean, cV_phase_std, reference='gaussian')

        # Shannon Entropy
        cV_shannon_mag = shannon_entropy(p_cV_hist_mag)
        cV_shannon_phase = shannon_entropy(p_cV_hist_phase)

        # cD:
        # Compute the probability distributions for magnitude and phase coefficients
        p_cD_hist_mag, bin_cD_edges_mag = np.histogram(cD_mag, bins=100, density=True)
        p_cD_hist_phase, bin_cD_edges_phase = np.histogram(cD_phase, bins=100, density=True)

        # Mean and standard deviation for KL divergence (for magnitude and phase)
        cD_mag_mean, cD_mag_std = np.mean(cD_mag), np.std(cD_mag)
        cD_phase_mean, cD_phase_std = np.mean(cD_phase), np.std(cD_phase)

        # KL Divergence
        cD_kl_mag = kl_divergence(p_cD_hist_mag, bin_cD_edges_mag, cD_mag_mean, cD_mag_std, reference='gaussian')
        cD_kl_phase = kl_divergence(p_cD_hist_phase, bin_cD_edges_phase, cD_phase_mean, cD_phase_std, reference='gaussian')

        # Shannon Entropy
        cD_shannon_mag = shannon_entropy(p_cD_hist_mag)
        cD_shannon_phase = shannon_entropy(p_cD_hist_phase)

        information_theoretic[f'level_{num_levels}'] = {
        'cH_KL_magnitude': cH_kl_mag,
        'cH_KL_phase': cH_kl_phase,
        'cH_shannon_magnitude': cH_shannon_mag,
        'cH_shannon_phase': cH_shannon_phase,
        'cV_KL_magnitude': cV_kl_mag,
        'cV_KL_phase': cV_kl_phase,
        'cV_shannon_magnitude': cV_shannon_mag,
        'cV_shannon_phase': cV_shannon_phase,
        'cD_KL_magnitude': cD_kl_mag,
        'cD_KL_phase': cD_kl_phase,
        'cD_shannon_magnitude': cD_shannon_mag,
        'cD_shannon_phase': cD_shannon_phase
        }

    # Generate all combinations of levels (ignoring level 0 for cA)
    level_combinations = itertools.combinations(range(1, num_levels + 1), 2)

    for level1, level2 in level_combinations:
        (cH_mag1, cH_phase1), (cV_mag1, cV_phase1), (cD_mag1, cD_phase1) = (mag[-level1], phase[-level1])
        (cH_mag2, cH_phase2), (cV_mag2, cV_phase2), (cD_mag2, cD_phase2) = (mag[-level2], phase[-level2])

        # cH:
        # Mutual Information and Conditional Entropy between levels for magnitude and phase
        cH_mi_mag = mutual_information(cH_mag1, cH_mag2)
        cH_mi_phase = mutual_information(cH_phase1, cH_phase2)

        cH_ce_mag = conditional_entropy(cH_mag2, cH_mag1)
        cH_ce_phase = conditional_entropy(cH_phase2, cH_phase1)

        # cV:
        # Mutual Information and Conditional Entropy between levels for magnitude and phase
        cV_mi_mag = mutual_information(cV_mag1, cV_mag2)
        cV_mi_phase = mutual_information(cV_phase1, cV_phase2)

        cV_ce_mag = conditional_entropy(cV_mag2, cV_mag1)
        cV_ce_phase = conditional_entropy(cV_phase2, cV_phase1)

        # cD:
        # Mutual Information and Conditional Entropy between levels for magnitude and phase
        cD_mi_mag = mutual_information(cD_mag1, cD_mag2)
        cD_mi_phase = mutual_information(cD_phase1, cD_phase2)

        cD_ce_mag = conditional_entropy(cD_mag2, cD_mag1)
        cD_ce_phase = conditional_entropy(cD_phase2, cD_phase1)

        information_theoretic[f'level_{level1}_to_{level2}'] = {
        'cH_MI_magitude': cH_mi_mag,
        'cH_MI_phase': cH_mi_phase,
        'cH_CE_magnitude': cH_ce_mag,
        'cH_CE_phase': cH_ce_phase,
        'cV_MI_magnitude': cV_mi_mag,
        'cV_MI_phase': cV_mi_phase,
        'cV_CE_magnitude': cV_ce_mag,
        'cV_CE_phase': cV_ce_phase,
        'cD_MI_magnitude': cD_mi_mag,
        'cD_MI_phase': cD_mi_phase,
        'cD_CE_magnitude': cD_ce_mag,
        'cD_CE_phase': cD_ce_phase
        }


# ============================================================
#                      Coherence Features
# ============================================================
    
def wavelet_cross_corelation(mag_c1, mag_c2):
    # Compute mean and standard deviation of the real parts
    mean_c1 = np.mean(mag_c1)
    mean_c2 = np.mean(mag_c2)
    std_c1 = np.std(mag_c1)
    std_c2 = np.std(mag_c2)

    # Cross-correlation for the real parts
    cross_correlation_real = np.sum((mag_c1 - mean_c1) * (mag_c2 - mean_c2)) / ((len(mag_c1) - 1) * std_c1 * std_c2)

    return cross_correlation_real


def complex_coherence(C1, C2):
    """
    Calculate the complex coherence between two sets of wavelet coefficients.
    
    Args:
    mag1 (np.array): Magnitude coefficients for signal 1.
    phase1 (np.array): Phase coefficients for signal 1.
    mag2 (np.array): Magnitude coefficients for signal 2.
    phase2 (np.array): Phase coefficients for signal 2.

    Returns:
    float: Complex coherence between the two signals.
    """

    # Compute the cross-spectral density (CSD)
    csd = np.mean(C1 * np.conj(C2))

    # Compute the auto-spectral densities (ASD) for both signals
    asd1 = np.mean(C1 * np.conj(C1))
    asd2 = np.mean(C2 * np.conj(C2))

    # Calculate coherence as the normalized cross-spectral density
    coherence = np.abs(csd) ** 2 / (asd1 * asd2)

    return coherence


def calculate_wavelet_cross_correlation(coeffs1, coeffs2, num_levels, type = ''):
    """
    Calculate coherence-based features for wavelet coefficients between two signals using complex coherence.
    
    Args:
    coeffs1 (list): List of wavelet coefficients for signal 1 where the first element is the approximation coefficients (cA)
                    and the rest are tuples of detail coefficients (cH, cV, cD) for each level.
                    Each element is a tuple containing (magnitude, phase).
    coeffs2 (list): List of wavelet coefficients for signal 2, structured similarly to coeffs1.
    num_levels (int): Number of wavelet decomposition levels.

    Returns:
    dict: Dictionary with coherence-based features (complex coherence) for each level, structured by coefficient type (cA, cH, cV, cD).
    """
    # Ensure both sets of coefficients have the same length
    if len (coeffs1) != len(coeffs2):
        raise ValueError("The wavelet coefficient arrays must have the same length.")

    coherence_features = {}

    coeffs1_mag, coeffs1_phase = coeffs1
    coeffs2_mag, coeffs2_phase = coeffs2

    # Process the approximation coefficients (cA) at the nth level
    cA_mag1, cA_phase1 = coeffs1_mag[0], coeffs1_phase[0]
    cA_mag2, cA_phase2 = coeffs2_mag[0], coeffs2_phase[0]

    cA1_complex = make_complex(cA_mag1, cA_phase1)
    cA2_complex = make_complex(cA_mag2, cA_phase2)

    # Calculate complex coherence for cA (approximation coefficients)
    coherence_cA = complex_coherence(cA1_complex, cA2_complex)

    # Calculate cross correlation between magnitude
    cross_correlation_cA = wavelet_cross_corelation(cA_mag1, cA_mag2)

    coherence_features[f'level_{num_levels}'] = {
        f'cA_{type}_complex_coherence': coherence_cA,
        f'cA_{type}_cross_correlation': cross_correlation_cA
    }

    for level in range(1, num_levels+1):
        (cH_mag1, cH_phase1), (cV_mag1, cV_phase1), (cD_mag1, cD_phase1) = (coeffs1_mag[-level], coeffs1_phase[-level])
        (cH_mag2, cH_phase2), (cV_mag2, cV_phase2), (cD_mag2, cD_phase2) = (coeffs2_mag[-level], coeffs2_phase[-level])

        cH1_complex = make_complex(cH_mag1, cH_phase1)
        cH2_complex = make_complex(cH_mag2, cH_phase2)

        cV1_complex = make_complex(cV_mag1, cV_phase1)
        cV2_complex = make_complex(cV_mag2, cV_phase2)

        cD1_complex = make_complex(cD_mag1, cD_phase1)
        cD2_complex = make_complex(cD_mag2, cD_phase2)

        # Calculate complex coherence for each coefficient type (cH, cV, cD)
        coherence_H = complex_coherence(cH1_complex, cH2_complex)
        coherence_V = complex_coherence(cV1_complex, cV2_complex)
        coherence_D = complex_coherence(cD1_complex, cD2_complex)

        # Calculate cross correlation for each coefficient type (cH, cV, cD)
        cc_H = wavelet_cross_corelation(cH_mag1, cH_mag2)
        cc_V = wavelet_cross_corelation(cV_mag1, cV_mag2)
        cc_D = wavelet_cross_corelation(cD_mag1, cD_mag2)

        coherence_features[f'level_{level}'] == {
            f'cH_{type}_complex_coherence': coherence_H,
            f'cV_{type}_complex_coherence': coherence_V,
            f'cD_{type}_complex_coherence': coherence_D,
            f'cH_{type}_cross_correlation': cc_H,
            f'cV_{type}_cross_correlation': cc_V,
            f'cD_{type}_cross_correlation': cc_D
        }

    return coherence_features


# Create
def calculate_inter_frequency_coherence(coeffs, num_levels):
    """
    Calculate inter-frequency coherence between different levels of the same signal.
    The approximation coefficients from level n are compared with the detail coefficients from level n-1.
    
    Args:
    coeffs (list): List of wavelet coefficients where the first element is the approximation coefficients (cA)
                   and the rest are tuples of detail coefficients (cH, cV, cD) for each level.
    num_levels (int): Number of wavelet decomposition levels.

    Returns:
    dict: Dictionary with inter-frequency coherence features for each level.
    """
    coherence_features = {}

    # Generate all combinations of levels (ignoring level 0 for cA)
    level_combinations = itertools.combinations(range(1, num_levels + 1), 2)

    coeffs_mag, coeffs_phase = coeffs

    for level1, level2 in level_combinations:
        (cH_mag1, cH_phase1), (cV_mag1, cV_phase1), (cD_mag1, cD_phase1) = (coeffs_mag[-level1], coeffs_phase[-level2])
        (cH_mag2, cH_phase2), (cV_mag2, cV_phase2), (cD_mag2, cD_phase2) = (coeffs_mag[-level2], coeffs_phase[-level2])

        cH1_complex = make_complex(cH_mag1, cH_phase1)
        cV1_complex = make_complex(cV_mag1, cV_phase1)
        cD1_complex = make_complex(cD_mag1, cD_phase1)

        cH2_complex = make_complex(cH_mag2, cH_phase2)
        cV2_complex = make_complex(cV_mag2, cV_phase2)
        cD2_complex = make_complex(cD_mag2, cD_phase2)

        # Calculate coherence for horizontal detail coefficients (cH)
        coherence_H = wavelet_cross_corelation(cH_mag1, cH_mag2)
        cc_H = complex_coherence(cH1_complex, cH2_complex)
        coherence_features[f'cH_level_{level1}_to_level_{level2}'] = coherence_H

        # Calculate coherence for vertical detail coefficients (cV)
        coherence_V = wavelet_cross_corelation(cV_mag1, cV_mag2)
        cc_V = complex_coherence(cV1_complex, cV2_complex)
        coherence_features[f'cV_level_{level1}_to_level_{level2}'] = coherence_V

        # Calculate coherence for diagonal detail coefficients (cD)
        coherence_D = wavelet_cross_corelation(cD_mag1, cD_mag2)
        cc_D = complex_coherence(cD1_complex, cD2_complex)
        coherence_features[f'cD_level_{level1}_to_level_{level2}'] = coherence_D

        coherence_features[f'level_{level1}_{level2}'] == {
            f'cH_complex_coherence': coherence_H,
            f'cV_complex_coherence': coherence_V,
            f'cD_complex_coherence': coherence_D,
            f'cH_cross_correlation': cc_H,
            f'cV_cross_correlation': cc_V,
            f'cD_cross_correlation': cc_D
        }

    return coherence_features


# ============================================================
#            Process Coeffs of single block - returned
# ============================================================

def process_wavelet_coeffs_block(coeffs, analysis, num_levels, coeffs2=None, coeffs_next_batch=None):
    """
    Process wavelet coefficients of a signal defined in a block of data size [Pages, Samples].
    
    signal_block:
    Numpy signal block to analyze.
    
    analysis:
    List of analysis types to run over the block.

    Num_levels:
    Number of levels for the analysis. 
    
    Returns:
    Map of dictionaries with the pairs (analysis_type, analysis), assuming 3 levels, each with the structure 
    {signal_index: {level_1: features, level_2: features, level_3: features}}.
    """

     # Initialize the dictionary to store the final features for each analysis type
    all_features = {}

    print("my analysis is: ", analysis, " ------- ", "number of levels is: ", num_levels)

    # Perform each type of analysis as specified
    if 'statistical' in analysis:
        statistical_features = calculate_basic_statistical_features(coeffs, num_levels)
        # print(statistical_features)
        # import time
        # time.sleep(5)
        all_features['statistical'] = statistical_features
    
    if 'energy' in analysis:
        energy_features = calculate_energy_based_features(coeffs, num_levels)
        all_features['energy'] = energy_features

    if 'frequency' in analysis:
        frequency_features = calculate_frequency_features(coeffs, num_levels)
        all_features['frequency'] = frequency_features

    if 'temporal' in analysis:
        temporal_features = calculate_temporal_features(coeffs, num_levels)
        all_features['temporal'] = temporal_features
    
    if 'spacial_coherence' in analysis:
        if coeffs2 == None:
            raise ValueError("Another coeff vec needed.")
        spacial_coherence_features = calculate_wavelet_cross_correlation(coeffs, coeffs2, num_levels, 'spacial')
        all_features['spacial_coherence'] = spacial_coherence_features

    if 'temporal_coherence' in analysis:
        if coeffs2 == None:
            raise ValueError("Another coeff vec needed.")
        temporal_coherence_features = calculate_wavelet_cross_correlation(coeffs, coeffs_next_batch, num_levels, 'temporal')
        all_features['temporal_coherence'] = temporal_coherence_features
    
    if 'inter_coherence' in analysis:
        inter_coherence_features = calculate_inter_frequency_coherence(coeffs, num_levels)
        all_features['inter_coherence'] = inter_coherence_features
    
    if 'advanced_statistical' in analysis:
        advanced_statistical_features = calculate_advanced_statistical_features(coeffs, num_levels)
        all_features['advanced_statistical'] = advanced_statistical_features

    if 'information_theoretic' in analysis:
        information_theoretic_features = calculate_information_theoretic_features(coeffs, num_levels)
        all_features['information_theoretic'] = information_theoretic_features
    
    return all_features


# ============================================================ #
# ============================================================ #

def main():
    parser = argparse.ArgumentParser(description="Choose the type of analysis.")

    parser.add_argument('--raw-data-folder', required=True, 
                        help="Name of the raw data folder inside 'results/1/'")
    
    parser.add_argument('--analysis-type', required=True, 
                        help="Name of the analysis type/folder inside 'results/1/'")
    
    parser.add_argument('--feature-family', choices=['statistical', 'energy', 'temporal', 'frequency', 
                                                     'information_theoretic', 'fractal', 'coherence', 
                                                     'advanced_statistical'], 
                                                     required=True, help="Type of features to extract")
    
    parser.add_argument('--single', action='store_true', help="Feature extraction from a SINGLE delta.")
    parser.add_argument('--level', type=int, required=True, help="Analysis level")

    args, unknown = parser.parse_known_args()

    if args.single:
        parser.add_argument('--batch', type=int, required=True, help="Batch to analyze from")
        parser.add_argument('--block-start', type=int, required=True, help="Initial page of the block to analyze from") 
        parser.add_argument('--block-size', type=int, required=True, help="Size of the block to analyze from")

    args = parser.parse_args()
    
    # base_dir = r'C:\Users\jeries\Desktop\thesis\results\1'
    base_dir = '/Volumes/Extreme SSD/thesis/runs/mixed'
    target_folder = os.path.join(base_dir, args.raw_data_folder, args.analysis_type, 'data')

    # if args.single:
    #     print("single")
    #     process_wavelet_coefficients_single_block(target_folder, args.feature_family, args.batch, args.block_start, args.block_size, args.level)
    # else:
    #     process_wavelet_coefficients(target_folder, args.feature_family, args.level)


if __name__ == "__main__":
    main()