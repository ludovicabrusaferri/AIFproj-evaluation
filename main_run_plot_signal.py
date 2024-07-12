import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
import random

random.seed(43)

index_to_subject = {
    0: 1,
    1: 2,
    2: 14,
    3: 37,
    4: 41,
    5: 42,
    6: 47,
    7: 69,
    8: 90,
    9: 72,
    10: 73
}

bool_value = 'kcl'  # or 'kcl'
studytype = 'AIF'

if bool_value == 'harvard':
    subjects = [5]
elif bool_value == 'kcl':
    indices = list(index_to_subject.keys())
    subjects = [index_to_subject[idx] for idx in indices]
else:
    raise ValueError("Invalid value for bool_value. Expected 'harvard' or 'kcl'.")

# Base paths
base_path_harvard = f'/Users/e410377/Desktop/Ludo/AlexLudo/ReformattedAIDataMarco/RESULTSNEW/metabolite_corrector_aif/test/{studytype}/'
base_path_harvard_true = '/Users/e410377/Desktop/Ludo/AlexLudo/ReformattedAIDataMarco/patient_data/metabolite_corrected_signal_data/'

base_path_kcl = f'/Users/e410377/Desktop/AIFproj-evaluation/OUT/ReformattedOriginalDataKCL_ALL/AIF/test/'  # Update with the actual base path for KCL
base_path_kcl_true = '/Users/e410377/Desktop/AIFproj-evaluation/OUT/ReformattedOriginalDataKCL_ALL/data_split/test/signal_data.pkl'  # Update with the actual true value path for KCL
base_path_kcl_true_orig = '/Users/e410377/Desktop/Ludo/AlexLudo/ReformattedOriginalDataKCL_ALL/patient_data/metabolite_corrected_signal_data/'

# Loop over all subjects
for index, subject in enumerate(subjects):
    if bool_value == 'harvard':
        signal_mean_path = base_path_harvard + f'{subject}_mean.pkl'
        signal_std_path = base_path_harvard + f'{subject}_stddev.pkl'
        output_file_path = base_path_harvard + f'{subject}_mean.txt'
        true_value_file_path = base_path_harvard_true + f'{subject}.txt'
        preprocessor_path = base_path_kcl + 'preprocessor.pkl'
    elif bool_value == 'kcl':
        signal_mean_path = base_path_kcl + f'{index}_mean.pkl'
        signal_std_path = base_path_kcl + f'{index}_stddev.pkl'
        output_file_path = base_path_kcl + f'{subject}_mean.txt'
        true_value_file_path_orig = base_path_kcl_true_orig + f'{subject}.txt'
        preprocessor_path = base_path_kcl + 'preprocessor.pkl'
        # Open the true value file in read binary mode ('rb') and extract the true values for the given index
        with open(base_path_kcl_true, 'rb') as file:
            true_value_data = pickle.load(file)
            true_values = true_value_data[index, 1, :]  # Corrected to take the 3rd column (index 2)

    # Check if the file paths exist
    if not (os.path.exists(signal_mean_path) and os.path.exists(signal_std_path)):
        print(f"One or more signal file paths do not exist for subject {subject}.")
        continue

    # Open the signal files in read binary mode ('rb')
    with open(signal_mean_path, 'rb') as file:
        signal_mean = pickle.load(file)

    with open(signal_std_path, 'rb') as file:
        signal_std = pickle.load(file)

    with open(preprocessor_path, "rb") as file:
        preprocessor = pickle.load(file)


    # Assuming signal_mean and signal_std are numpy arrays
    signal_samples = []
    for i in range(int((np.max(signal_std) * 31.0) + 1.0)):
        signal_sample = np.random.normal(loc=signal_mean, scale=signal_std)
        signal_sample = np.reshape(preprocessor.inverse_transform(np.reshape(signal_sample, (-1, 1))), signal_sample.shape)
        signal_samples.append(signal_sample)

    signal = np.mean(signal_samples, axis=0)
    signal = np.squeeze(signal)

    # Load true values for Harvard
    if bool_value == 'harvard':
        true_values = np.loadtxt(true_value_file_path)

    # Load original true values for KCL
    if bool_value == 'kcl':
        true_values_orig = np.loadtxt(true_value_file_path_orig)

    # Print signal into a text file
    with open(output_file_path, 'w') as output_file:
        for value in signal:
            output_file.write(str(value) + '\n')

    # Plot both signals
    plt.plot(signal, label='Generated Signal')
    plt.plot(true_values, label='True Value', linestyle='--', linewidth=2)  # Thicker line for True Value

    # Plot original true values for KCL if applicable
    if bool_value == 'kcl':
        plt.plot(true_values_orig, label='Original True Value', linestyle='--', alpha=0.7)

    plt.title(f'Signal Comparison for Subject {subject}')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_file_path = output_file_path.replace('_mean.txt', '_plot.png')
    plt.savefig(plot_file_path)
    plt.close()

    print(f"Processed and saved data for subject {subject}")
