import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt

subject = 73
# Define the variable
bool = 'kcl'  # or 'harvard'
index = 10

# Base paths
base_path_harvard = f'/Users/e410377/Desktop/Ludo/AlexLudo/ReformattedAIDataMarco/RESULTSNEW/metabolite_corrector_aif/test/{subject}/'
base_path_harvard_true = '/Users/e410377/Desktop/Ludo/AlexLudo/ReformattedAIDataMarco/patient_data/metabolite_corrected_signal_data/'

base_path_kcl = f'/Users/e410377/Desktop/AIFproj-evaluation/OUT/ReformattedOriginalDataKCL_ALL/AIF/test/'  # Update with the actual base path for KCL
base_path_kcl_true = '/Users/e410377/Desktop/AIFproj-evaluation/OUT/ReformattedOriginalDataKCL_ALL/data_split/test/signal_data.pkl'  # Update with the actual true value path for KCL
base_path_kcl_true_orig = '/Users/e410377/Desktop/Ludo/AlexLudo/ReformattedOriginalDataKCL_ALL/patient_data/metabolite_corrected_signal_data/'

# Set paths based on the value of 'bool'
if bool == 'harvard':
    signal_mean_path = base_path_harvard + f'{subject}_mean.pkl'
    signal_std_path = base_path_harvard + f'{subject}_stddev.pkl'
    output_file_path = base_path_harvard + f'{subject}_mean.txt'
    true_value_file_path = base_path_harvard_true + f'{subject}.txt'
elif bool == 'kcl':
    signal_mean_path = base_path_kcl + f'{index}_mean.pkl'
    signal_std_path = base_path_kcl + f'{index}_stddev.pkl'
    output_file_path = base_path_kcl + f'{subject}_mean.txt'
    true_value_file_path_orig = base_path_kcl_true_orig + f'{subject}.txt'
    # Open the true value file in read binary mode ('rb') and extract the true values for the given index
    with open(base_path_kcl_true, 'rb') as file:
        true_value_data = pickle.load(file)
        true_values = true_value_data[index, 1, :]  # Corrected to take the 3rd column (index 2)
else:
    raise ValueError("Invalid value for bool. Expected 'harvard' or 'kcl'.")

# You can now use these paths in your code
print(f"Signal mean path: {signal_mean_path}")
print(f"Signal std path: {signal_std_path}")
print(f"Output file path: {output_file_path}")

# Check if the file paths exist
if os.path.exists(signal_mean_path) and os.path.exists(signal_std_path):
    print("Signal file paths exist.")
else:
    print("One or more signal file paths do not exist.")

# Open the signal files in read binary mode ('rb')
with open(signal_mean_path, 'rb') as file:
    signal_mean = pickle.load(file)

with open(signal_std_path, 'rb') as file:
    signal_std = pickle.load(file)

# Assuming signal_mean and signal_std are numpy arrays
signal_samples = []
for i in range(int((np.max(signal_std) * 31.0) + 1.0)):
    signal_samples.append(np.random.normal(loc=signal_mean, scale=signal_std))

signal = np.mean(signal_samples, axis=0)
signal = np.squeeze(signal)

# Load true values for Harvard
if bool == 'harvard':
    true_values = np.loadtxt(true_value_file_path)

# Load original true values for KCL
if bool == 'kcl':
    true_values_orig = np.loadtxt(true_value_file_path_orig)

# Print signal into a text file
with open(output_file_path, 'w') as output_file:
    for value in signal:
        output_file.write(str(value) + '\n')

# Plot both signals
plt.plot(signal, label='Generated Signal')
plt.plot(true_values, label='True Value', linestyle='--', linewidth=2)  # Thicker line for True Value

# Plot original true values for KCL if applicable
if bool == 'kcl':
    plt.plot(true_values_orig, label='Original True Value', linestyle='--', alpha=0.7)

plt.title('Signal Comparison')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
