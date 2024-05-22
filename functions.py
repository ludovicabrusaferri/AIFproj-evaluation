import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.integrate import cumtrapz
from sklearn.linear_model import LinearRegression
import seaborn as sns


def run_evaluation(project_dir, num_subjects, truemethod, predictedmethod):
    folders1 = os.path.join(project_dir, 'VtsOUT', truemethod)
    folders2 = os.path.join(project_dir, 'VtsOUT', predictedmethod)

    files1 = [f for f in os.listdir(folders1) if f.endswith('.txt')]
    num_files_per_subject = len(files1) // num_subjects
    num_ROIs = num_files_per_subject

    data_matrix_folder1 = np.zeros((num_subjects, num_ROIs))
    data_matrix_folder2 = np.zeros((num_subjects, num_ROIs))

    for subj_index in range(num_subjects):
        current_subject_files1 = [f for f in os.listdir(folders1) if f.startswith(f'{subj_index}_')]
        #current_subject_files2 = [f for f in os.listdir(folders2) if f.startswith(f'{subj_index}_')]

        filenames = [os.path.splitext(f)[0] for f in current_subject_files1]
        if len(filenames) != len(set(filenames)):
            print(f'Duplicate filenames found for subject {subj_index}')
            # Handle duplicates as needed

        for file_index, filename in enumerate(current_subject_files1):
            print(f'Assigning filename "{filename}" to column {file_index}')
            data = np.loadtxt(os.path.join(folders1, filename))
            data_matrix_folder1[subj_index, file_index] = data

            data = np.loadtxt(os.path.join(folders2, filename))
            data_matrix_folder2[subj_index, file_index] = data

    data_sum_1 = np.mean(data_matrix_folder1, axis=1)
    data_sum_2 = np.mean(data_matrix_folder2, axis=1)

    plt.figure(figsize=(8, 6))
    plt.scatter(data_sum_1, data_sum_2, color='r', s=30)

    # Regression line
    coeffs = np.polyfit(data_sum_1, data_sum_2, 1)
    regression_line = np.poly1d(coeffs)
    plt.plot(data_sum_1, regression_line(data_sum_1), color='blue')

    # Identity line
    plt.plot(data_sum_1, data_sum_1, linestyle='--', color='gray')

    # Regression coefficient
    correlation_coefficient = np.corrcoef(data_sum_1, data_sum_2)[0, 1]
    #plt.text(np.min(data_sum_1), np.max(data_sum_2), f'Rho: {correlation_coefficient:.2f}', fontsize=12)

    plt.title(f'Rho: {correlation_coefficient:.2f}; p: {pearsonr(data_sum_1, data_sum_2)[1]:.2f}')
    plt.xlabel('True Vt')
    plt.ylabel('Predicted Vt')
    plt.show()

def calculate_logan_vt(frame_time_filename, reference_filename, target_filename, tstar, output_filename, output_directory,
                       reference_name, plot_vt):
    time_all = np.loadtxt(frame_time_filename, delimiter=',')
    reference_tac = np.loadtxt(reference_filename)
    target_tac = np.loadtxt(target_filename)

    # Transpose if target is a row vector
    if target_tac.shape[0] == 1:
        target_tac = target_tac.T

    # Time calculations
    time = (time_all[:, 1] / 60 + time_all[:, 0] / 60) / 2
    dt = time_all[:, 1] / 60 - time_all[:, 0] / 60

    # Calculate integrated reference and target
    int_ref = cumtrapz(reference_tac, time, initial=0)
    int_target = cumtrapz(target_tac, time, initial=0)
    intercept = np.ones_like(reference_tac)

    X_matrix = np.column_stack([int_ref / target_tac, intercept])
    Y_vector = int_target / target_tac

    tstar_index = np.argmax(time >= tstar)
    X_selected = X_matrix[tstar_index:, :]
    Y_selected = Y_vector[tstar_index:]

    weight_matrix = np.diag(np.ones_like(dt))
    weight_matrix_selected = weight_matrix[tstar_index:, tstar_index:]

    regression_coefficients = np.linalg.lstsq(weight_matrix_selected @ X_selected, weight_matrix_selected @ Y_selected,
                                             rcond=None)[0]
    Vt = regression_coefficients[0]

    if np.isnan(Vt):
        raise ValueError('Vt is NaN. Check input values!')

    if plot_vt:
        plt.figure()
        plt.plot(X_matrix[:, 0], Y_vector, '*')
        plt.plot(X_selected[:, 0], X_selected @ regression_coefficients, 'k')
        plt.plot(X_matrix[tstar_index, 0], Y_vector[tstar_index], 'o', markersize=10)
        plt.title(f'Logan Vt: target-{reference_name}, reference-plasma, tstar: {tstar} minutes')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.show()

    np.savetxt(output_filename + '.txt', [Vt])
    return Vt


def run_simulate_data(true_input, predicted_input, num_subjects, noise_level):
    if not os.path.exists(predicted_input):
        os.makedirs(predicted_input)

    for index in range(num_subjects):
        # Create a subdirectory for each subject
        subject_directory = os.path.join(predicted_input, f'{index}')
        os.makedirs(subject_directory, exist_ok=True)

        input_filename = os.path.join(true_input, f'{index}.txt')
        original_signal = np.loadtxt(input_filename)

        noisy_signal = original_signal + noise_level * np.random.randn(*original_signal.shape)

        # Output files are saved in the subject-specific subdirectory
        output_filename_mean = os.path.join(subject_directory, f'{index}_mean.txt')
        np.savetxt(output_filename_mean, noisy_signal, delimiter='\t')

        output_filename_std = os.path.join(subject_directory, f'{index}_std.txt')
        np.savetxt(output_filename_std, np.ones_like(original_signal), delimiter='\t')

    print('Noisy signals with mean saved, and plotted.')

    t = 1
    plt.figure()
    for plot_index in np.random.choice(range(num_subjects), size=2, replace=False):
        # Load data from the subject-specific subdirectory
        subject_directory = os.path.join(predicted_input, f'{plot_index}')

        noisy_filename_mean = os.path.join(subject_directory, f'{plot_index}_mean.txt')
        noisy_signal_mean = np.loadtxt(noisy_filename_mean)

        std_filename = os.path.join(subject_directory, f'{plot_index}_std.txt')
        std_signal = np.loadtxt(std_filename)

        true_filename = os.path.join(true_input, f'{plot_index}.txt')
        true_signal = np.loadtxt(true_filename)

        plt.subplot(1, 2, t)
        plt.plot(noisy_signal_mean, '-r', linewidth=2)
        plt.plot(true_signal, '-b', linewidth=2)
        plt.title(f'Noisy Signal subj{plot_index}')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.grid(True)
        t += 1

    plt.show()


def run_calculate_VT(method, true_input, predicted_input, output_directory, petframestartstop, tac_directory, subjects,
                     plot_vt):
    for i in subjects:
        print(f'============ SUBJ {i} and i: {i} \n\n =======\n')
        tstar = 15
        frame_time_fn = os.path.join(petframestartstop, f'{i}.txt')

        # Use a different variable for the loop iteration
        current_predicted_input = os.path.join(predicted_input, f'{i}')

        if method == 'AIF':
            plasma_lin_filename = os.path.join(current_predicted_input, f'{i}_mean.txt')
        elif method == 'TRUE':
            plasma_lin_filename = os.path.join(true_input, f'{i}.txt')

        # Check if plasma_lin_filename exists
        if not os.path.exists(plasma_lin_filename):
            print(f'Skipping subject {i} because plasma_lin_filename {plasma_lin_filename} does not exist.')
            continue

        tacs_directory = os.path.join(tac_directory, str(i))
        tacs = [f for f in os.listdir(tacs_directory) if f.startswith(f'{i}_') and f.endswith('.txt')]

        for target_fn in tacs:
            tac_name, _ = os.path.splitext(target_fn)
            reference_fn = plasma_lin_filename

            target_path = os.path.join(tacs_directory, target_fn)
            output_fn = os.path.join(output_directory, f'{tac_name}_Logan_Vt_{tstar}.txt')

            if not os.path.exists(output_fn):
                vt = calculate_logan_vt(frame_time_fn, reference_fn, target_path, tstar, output_fn, output_directory,
                                        reference_fn, plot_vt)

                print(f'vt={vt}')
                if vt < 0:
                    raise ValueError('vt cannot be negative.')


def plot_correlation_with_regression_line(subject, path1, path2):
    # Function to read a single float value from each file
    def read_data_files(path, subject):
        data = []
        for file_name in os.listdir(path):
            if file_name.startswith(f'{subject}_') and file_name.endswith('.txt'):
                file_path = os.path.join(path, file_name)
                with open(file_path, 'r') as file:
                    value = float(file.read().strip())
                    data.append(value)
        return data

    # Read data from both directories
    data1 = read_data_files(path1, subject)
    data2 = read_data_files(path2, subject)

    # Check if we have at least one file from each directory
    if not data1 or not data2:
        raise ValueError("Not enough data files found to plot correlation.")

    # Convert lists to NumPy arrays
    x = np.array(data1)
    y = np.array(data2)

    # Plotting scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x, y=y)

    # Adding regression line
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))
    plt.plot(x, y_pred, color='red', linewidth=2)

    # Plotting identity line
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--', linewidth=2)

    # Equation of the regression line
    slope = model.coef_[0]
    intercept = model.intercept_
    equation = f'y = {slope:.2f}x + {intercept:.2f}'

    # Adding equation to the plot
    plt.text(0.5, 0.9, equation, fontsize=12, transform=plt.gca().transAxes)

    # Adding labels and title
    plt.xlabel('Values from first data set')
    plt.ylabel('Values from second data set')
    plt.title(f'Scatter plot with regression line for subject {subject}')

    plt.show()
