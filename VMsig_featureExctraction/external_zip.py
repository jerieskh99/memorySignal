import zipfile 
import os
import numpy as np


def zip_dir(p_source_dir, P_output_dir):
    """
    Zip the entire directory, including all subdirectories and files.
    
    :param p_source_dir: Path of the directory to zip.
    :param P_output_dir: Path where the zip file will be saved.
    """

    print("p_source_dir - ", p_source_dir)
    print("P_output_dir - ", P_output_dir)
    with zipfile.ZipFile(P_output_dir, 'w', zipfile.ZIP_DEFLATED) as zip_f:
        for root, dirs, files in os.walk(p_source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zip_f.write(file_path, os.path.relpath(file_path, p_source_dir))

    print(f"Zipping {p_source_dir} completed. Saved to: {P_output_dir}")


def unzip_dir(zip_file_path, output_dir):
    """
    Unzip the specified zip file into the target directory.
    
    :param zip_file_path: Path of the zip file to unzip.
    :param output_dir: Directory where the contents will be extracted.
    """
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print(f"Unzipping {zip_file_path} completed. Extracted to: {output_dir}")




def load_coefficients_from_zip(zip_file_path):
    real_coeffs = []
    imag_coeffs = []
    
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        for filename in z.namelist():
            if 'real' in filename:
                with z.open(filename) as file:
                    real_coeffs.append(np.loadtxt(file))
            elif 'imag' in filename:
                with z.open(filename) as file:
                    imag_coeffs.append(np.loadtxt(file))
                    
    # Flatten and combine real and imaginary parts
    real_coeffs = np.array(real_coeffs).flatten()
    imag_coeffs = np.array(imag_coeffs).flatten()
    combined_coeffs = np.hstack((real_coeffs, imag_coeffs))
    
    return combined_coeffs