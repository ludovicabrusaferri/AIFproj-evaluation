import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt

signal_mean_path = '/Users/e410377/Desktop/Ludo/AlexLudo/ReformattedAIDataMarco/RESULTS/metabolite_corrector_idif/test/test/15_mean.pkl'
signal_std_path = '/Users/e410377/Desktop/Ludo/AlexLudo/ReformattedAIDataMarco/RESULTS/metabolite_corrector_idif/test/test/15_stddev.pkl'
standard_scaler_path = '/Users/e410377/Desktop/Ludo/AlexLudo/ReformattedAIDataMarco/RESULTS/metabolite_corrector_idif/standard_scaler.pkl'
output_file_path = '/Users/e410377/Desktop/Ludo/AlexLudo/ReformattedAIDataMarco/RESULTS/metabolite_corrector_idif/test/15/15_mean.txt'
true_value_file_path = '/Users/e410377/Desktop/Ludo/AlexLudo/ReformattedAIDataMarco/patient_data/metabolite_corrected_signal_data/15.txt'
time_activity_curves_path = '/Users/e410377/Desktop/Ludo/AlexLudo/ReformattedAIDataMarco/patient_data/time_activity_curves'

# Check if the file paths exist
if os.path.exists(signal_mean_path) and os.path.exists(signal_std_path) and os.path.exists(standard_scaler_path) and os.path.exists(true_value_file_path):
    print("All file paths exist.")
else:
    print("One or more file paths do not exist.")

# Open the files in read binary mode ('rb')
with open(signal_mean_path, 'rb') as file:
    signal_mean = pickle.load(file)

with open(signal_std_path, 'rb') as file:
    signal_std = pickle.load(file)

with open(standard_scaler_path, 'rb') as file:
    standard_scaler = pickle.load(file)

signal_mean = signal_mean.numpy()
signal_std = signal_std.numpy()

signal_samples = []
for i in range(int((np.max(signal_std) * 31.0) + 1.0)):
    signal_samples.append(np.random.normal(loc=signal_mean, scale=signal_std))

signal = np.mean(signal_samples, axis=0)

signal = np.reshape(standard_scaler.inverse_transform(np.reshape(signal, (-1, 1))), signal.shape)

# Load true values
true_values = np.loadtxt(true_value_file_path)

# Print signal into a text file
with open(output_file_path, 'w') as output_file:
    for value in signal:
        output_file.write(str(value) + '\n')

# Plot both signals
plt.plot(signal, label='Generated Signal')
plt.plot(true_values, label='True Value', linestyle='--')
plt.title('Signal Comparison')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
