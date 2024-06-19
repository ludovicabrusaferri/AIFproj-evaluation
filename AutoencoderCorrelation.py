import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler

subject = 69
num_timestamps = 26
flattened_signals = []

plotsave = '/Users/e410377/Desktop/AIFproj-evaluation/OUT/ReformattedOriginalDataKCL_ALL/figures'

# Load the true values
true_value_file_path = f'/Users/e410377/Desktop/Ludo/AlexLudo/ReformattedOriginalDataKCL_ALL/patient_data/metabolite_corrected_signal_data/{subject}.txt'
metuncor_path = f'/Users/e410377/Desktop/Ludo/AlexLudo/ReformattedOriginalDataKCL_ALL/patient_data/signal_data/{subject}.txt'

if os.path.exists(true_value_file_path):
    true_values = np.loadtxt(true_value_file_path)
    print("Loaded true values")
else:
    raise FileNotFoundError(f"True value file path does not exist: {true_value_file_path}")

if os.path.exists(metuncor_path):
    uncor_values = np.loadtxt(metuncor_path)
    print("Loaded uncorrected values")
else:
    raise FileNotFoundError(f"Uncorrected value file path does not exist: {metuncor_path}")

for timestamp in range(num_timestamps):
    signal_mean_path = f'/Users/e410377/Desktop/AIFproj-evaluation/OUT/ReformattedOriginalDataKCL_ALL/autoencoder/{subject}_0/{timestamp}_latent_mean.pkl'
    signal_std_path = f'/Users/e410377/Desktop/AIFproj-evaluation/OUT/ReformattedOriginalDataKCL_ALL/autoencoder/{subject}_0/{timestamp}_latent_stddev.pkl'

    # Check if the file paths exist
    if os.path.exists(signal_mean_path) and os.path.exists(signal_std_path):
        with open(signal_mean_path, 'rb') as file:
            signal_mean = pickle.load(file)

        with open(signal_std_path, 'rb') as file:
            signal_std = pickle.load(file)

        signal_mean = signal_mean.numpy()
        signal_std = signal_std.numpy()
        print(f"Loaded data for timestamp {timestamp}")

        signal_samples = []
        for i in range(int(np.max([np.ceil(np.max(signal_std) * 32.0), 1.0]))):
            signal_samples.append(np.random.normal(loc=signal_mean, scale=signal_std))

        signal = np.mean(signal_samples, axis=0)
        signal = np.squeeze(signal)

        # Flatten the array
        flattened_signal = signal.flatten()
        flattened_signals.append(flattened_signal)
    else:
        print(f"File paths for timestamp {timestamp} do not exist.")

print("Finished loading and processing all timestamps")

# Convert the list to a 26xN matrix
flattened_signals_matrix = np.array(flattened_signals)
print("Converted flattened signals to a matrix")

# Normalize flattened true signal
scaler_true = StandardScaler()
true_values_normalized = np.reshape(scaler_true.fit_transform(np.reshape(true_values, (-1, 1))), true_values.shape)
print("Normalized true values")

# Normalize uncorrected values
scaler_uncor = StandardScaler()
uncor_values_normalized = np.reshape(scaler_uncor.fit_transform(np.reshape(uncor_values, (-1, 1))), uncor_values.shape)
print("Normalized uncorrected values")

# Normalize flattened signals matrix
scaler_flattened = StandardScaler()
flattened_signals_matrix_normalized = np.reshape(scaler_flattened.fit_transform(np.reshape(flattened_signals_matrix, (-1, flattened_signals_matrix.shape[1]))), flattened_signals_matrix.shape)
print("Normalized flattened signals matrix")

# Plot the signals in a 5x6 grid
fig, axs = plt.subplots(5, 6, figsize=(20, 20))

for i in range(5):
    for j in range(6):
        timestamp_index = i * 6 + j
        if timestamp_index < num_timestamps:
            axs[i, j].plot(flattened_signals_matrix_normalized[timestamp_index])
            axs[i, j].set_title(f'Timestamp {timestamp_index}')
            axs[i, j].set_xlabel('Index')
            axs[i, j].set_ylabel('Value')

plt.tight_layout()

# Save the figure as a jpg file
fig_path = os.path.join(plotsave, f'subject_{subject}_flattened_signals_plot.jpg')
plt.savefig(fig_path, format='jpg')
plt.close()
print("Saved flattened signals plot")

# Flatten the normalized true values
flattened_true_values_normalized = true_values_normalized.flatten()
print("Flattened true values")

# Flatten the normalized uncorrected values
flattened_uncor_values_normalized = uncor_values_normalized.flatten()
print("Flattened uncorrected values")

# Calculate the correlation for each element across the timestamps
correlations = []
for i in range(flattened_signals_matrix_normalized.shape[1]):
    correlation = np.corrcoef(flattened_signals_matrix_normalized[:, i], flattened_true_values_normalized)[0, 1]
    correlations.append((i, correlation))  # Store index and correlation
    if i % 1000 == 0:
        print(f"Calculated correlation for element {i}")

# Sort correlations in descending order
correlations_sorted = sorted(correlations, key=lambda x: x[1], reverse=True)
print("Sorted correlations")

# Extract sorted indices and values
sorted_indices = [x[0] for x in correlations_sorted]
sorted_correlations = [x[1] for x in correlations_sorted]

# Convert correlations to a numpy array
sorted_correlations = np.array(sorted_correlations)

# Plot the best score vs true and uncorrected, the correlation, and original true vs uncorrected vs best score before rescaling
fig, axs = plt.subplots(1, 4, figsize=(32, 8))

best_index = sorted_indices[0]
best_signal_normalized = flattened_signals_matrix_normalized[:, best_index]
best_signal_original = flattened_signals_matrix[:, best_index]

# Plot the best signal vs the true signal and uncorrected signal (normalized)
axs[0].plot(best_signal_normalized, label='Best Signal')
axs[0].plot(flattened_true_values_normalized, label='True Signal', linestyle='--')
axs[0].plot(flattened_uncor_values_normalized, label='Uncorrected Signal', color='black', linestyle='-.')
axs[0].set_title(f'Subject {subject} - Best Signal vs True and Uncorrected Signal (Normalized)')
axs[0].set_xlabel('Index')
axs[0].set_ylabel('Value')
axs[0].legend()

# Create a scatter plot with a regression line (normalized)
slope, intercept, r_value, p_value, std_err = linregress(flattened_true_values_normalized, best_signal_normalized)
line = slope * flattened_true_values_normalized + intercept
axs[1].scatter(flattened_true_values_normalized, best_signal_normalized, label='Data Points')
axs[1].plot(flattened_true_values_normalized, line, color='red', label=f'Regression Line (r={r_value:.2f})')
axs[1].set_title(f'Subject {subject} - Scatter Plot with Regression Line (Normalized)')
axs[1].set_xlabel('True Signal')
axs[1].set_ylabel('Best Signal')
axs[1].legend()

# Plot the original true, uncorrected, and best signal before rescaling
axs[2].plot(true_values, label='True Signal')
axs[2].plot(uncor_values, label='Uncorrected Signal', color='black', linestyle='-.')
axs[2].plot(best_signal_original, label='Best Signal')
axs[2].set_title(f'Subject {subject} - Original True vs Uncorrected vs Best Signal (Original)')
axs[2].set_xlabel('Index')
axs[2].set_ylabel('Value')
axs[2].legend()

# Plot the rescaling parameters (mean and scale) for the best signal
axs[3].plot(scaler_flattened.mean_, label='Mean Value', linestyle='--')
axs[3].plot(scaler_flattened.scale_, label='Scale Value', linestyle='--')
axs[3].set_title(f'Subject {subject} - Rescaling Parameters for Best Signal')
axs[3].set_xlabel('Feature Index')
axs[3].set_ylabel('Value')
axs[3].set_ylim([-4, 4])
axs[3].legend()

# Save the plot as a jpg file
best_signal_plot_path = os.path.join(plotsave, f'subject_{subject}_best_signals_vs_true_uncorrected_rescaling.jpg')
plt.savefig(best_signal_plot_path, format='jpg')
plt.close()
print("Saved best signals vs true and uncorrected signal plot with rescaling parameters")
