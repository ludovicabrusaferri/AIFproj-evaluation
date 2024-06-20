import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler
import nibabel as nib

subject = 24
num_timestamps = 26

def load_data(true_value_file_path, metuncor_path):
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

    return true_values, uncor_values

def load_and_process_signals(subject, num_timestamps):
    flattened_signals = []
    original_shapes = []
    workdir = '/Users/e410377/Desktop/AIFproj-evaluation/OUT/ReformattedOriginalDataKCL_ALL/autoencoder'
    for timestamp in range(num_timestamps):
        signal_mean_path = f'{workdir}/{subject}_0/{timestamp}_latent_mean.pkl'
        signal_std_path = f'{workdir}/{subject}_0/{timestamp}_latent_stddev.pkl'

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

            original_shapes.append(signal.shape)
            flattened_signal = signal.flatten()
            flattened_signals.append(flattened_signal)
        else:
            print(f"File paths for timestamp {timestamp} do not exist.")

    return np.array(flattened_signals), original_shapes

def load_and_process_images(subject, num_timestamps):
    flattened_images = []
    workdir = '/Users/e410377/Desktop/Ludo/AlexLudo/ReformattedOriginalDataKCL_ALL/patient_data/image_data'
    for timestamp in range(num_timestamps):
        image_path = f'{workdir}/{subject}/{subject}_00{timestamp:02d}.nii.gz'

        if os.path.exists(image_path):
            img = nib.load(image_path)
            img_data = img.get_fdata()
            img_flattened = img_data.flatten()
            flattened_images.append(img_flattened)
            print(f"Loaded and flattened image data for timestamp {timestamp}")
        else:
            print(f"Image file path for timestamp {timestamp} does not exist.")

    return np.array(flattened_images)

def normalize_data(data):
    scaler = StandardScaler()
    return np.reshape(scaler.fit_transform(np.reshape(data, (-1, data.shape[1]))), data.shape)

def calculate_and_plot_correlations(data_matrix, flattened_true_values_normalized, flattened_uncor_values_normalized, data_type, plotsave):
    # Calculate the correlation for each element across the timestamps
    correlations = []
    for i in range(data_matrix.shape[1]):
        correlation = np.corrcoef(data_matrix[:, i], flattened_true_values_normalized)[0, 1]
        if not np.isnan(correlation):
            correlations.append((i, correlation))  # Store index and correlation
        if i % 1000 == 0:
            print(f"Calculated correlation for {data_type} element {i}: {correlation}")

    # Sort correlations in descending order
    correlations_sorted = sorted(correlations, key=lambda x: x[1], reverse=True)
    print(f"Sorted {data_type} correlations")

    # Extract sorted indices and values
    sorted_indices = [x[0] for x in correlations_sorted]
    sorted_correlations = [x[1] for x in correlations_sorted]

    # Plot the best score vs true and uncorrected, the correlation, original true vs uncorrected vs best score before rescaling,
    # and the rescaling parameters (mean and scale) for the best signal

    fig, axs = plt.subplots(2, 1, figsize=(20, 32))

    best_index = sorted_indices[0]
    best_signal_normalized = data_matrix[:, best_index]

    # Plot the best signal vs the true signal and uncorrected signal (normalized)
    axs[0].plot(best_signal_normalized, label=f'Best {data_type.capitalize()} Signal')
    axs[0].plot(flattened_true_values_normalized, label='True Signal', linestyle='--')
    axs[0].plot(flattened_uncor_values_normalized, label='Uncorrected Signal', color='black', linestyle='-.')
    axs[0].set_title(f'Subject {subject} - Best {data_type.capitalize()} Signal vs True and Uncorrected Signal (Normalized)')
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
    axs[1].set_ylabel(f'Best {data_type.capitalize()} Signal')
    axs[1].legend()

    # Save the plot as a jpg file
    plot_path = os.path.join(plotsave, f'subject_{subject}_best_{data_type}_signals_vs_true_uncorrected_rescaling.jpg')
    plt.savefig(plot_path, format='jpg')
    plt.close()
    print(f"Saved best {data_type} signals vs true and uncorrected signal plot with rescaling parameters")

def plot_hidden_layer_signals(flattened_signals_matrix_normalized, num_timestamps, plotsave):
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
    fig_path = os.path.join(plotsave, f'subject_{subject}_flattened_signals_plot.jpg')
    plt.savefig(fig_path, format='jpg')
    plt.close()
    print("Saved flattened signals plot")

def main():
    # Define file paths
    plotsave = '/Users/e410377/Desktop/AIFproj-evaluation/OUT/ReformattedOriginalDataKCL_ALL/figures'
    true_value_file_path = f'/Users/e410377/Desktop/Ludo/AlexLudo/ReformattedOriginalDataKCL_ALL/patient_data/metabolite_corrected_signal_data/{subject}.txt'
    metuncor_path = f'/Users/e410377/Desktop/Ludo/AlexLudo/ReformattedOriginalDataKCL_ALL/patient_data/signal_data/{subject}.txt'

    # Load true and uncorrected values
    true_values, uncor_values = load_data(true_value_file_path, metuncor_path)

    # Load and process signals
    flattened_signals_matrix, original_shapes = load_and_process_signals(subject, num_timestamps)
    print("Converted flattened signals to matrix")

    # Load and process images
    flattened_images_matrix = load_and_process_images(subject, num_timestamps)
    print("Converted flattened images to matrix")

    # Normalize data
    flattened_signals_matrix_normalized = normalize_data(flattened_signals_matrix)
    flattened_images_matrix_normalized = normalize_data(flattened_images_matrix)
    true_values_normalized = normalize_data(true_values.reshape(-1, 1)).flatten()
    uncor_values_normalized = normalize_data(uncor_values.reshape(-1, 1)).flatten()

    # Plot hidden layer signals
    plot_hidden_layer_signals(flattened_signals_matrix_normalized, num_timestamps, plotsave)

    # Plot image
    #plot_hidden_layer_signals(flattened_images_matrix_normalized, num_timestamps, plotsave)

    # Calculate and plot correlations for latent space signals
    calculate_and_plot_correlations(flattened_signals_matrix_normalized, true_values_normalized, uncor_values_normalized, 'latent', plotsave)


    # Calculate and plot correlations for images
    calculate_and_plot_correlations(flattened_images_matrix_normalized, true_values_normalized, uncor_values_normalized, 'image', plotsave)

if __name__ == "__main__":
    main()
