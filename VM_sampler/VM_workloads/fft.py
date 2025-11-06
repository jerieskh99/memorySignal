import numpy as np

# Define the size of the array (e.g., 2 million elements)
array_size = 80000

# Generate a large random array
large_array = np.random.rand(array_size)

iteration = 0

# Infinite loop to perform FFT
while True:
    print(f"Iteration {iteration}: FFT started")
    fft_result = np.fft.fft(large_array)
    print(f"Iteration {iteration}: FFT complete")
    iteration += 1
