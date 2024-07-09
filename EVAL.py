import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from scipy.integrate import cumtrapz


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

    try:
        regression_coefficients = np.linalg.lstsq(weight_matrix_selected @ X_selected, weight_matrix_selected @ Y_selected, rcond=None)[0]
        Vt = regression_coefficients[0]
    except np.linalg.LinAlgError:
        Vt = np.nan


    #if np.isnan(Vt):
     #   raise ValueError('Vt is NaN. Check input values!')

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


def run_calculate_VT(method, input, output_directory, petframestartstop, tac_directory, subjects,
                     plot_vt):
    for i in subjects:
        print(f'============ SUBJ {i} and i: {i} \n\n =======\n')
        tstar = 15
        frame_time_fn = os.path.join(petframestartstop, f'{i}.txt')

        # Use a different variable for the loop iteration
        current_predicted_input = os.path.join(input, f'{i}')

        if method == 'TRUE':
            plasma_lin_filename = os.path.join(input, f'{i}.txt')
        else:
            plasma_lin_filename = os.path.join(input, f'{i}_mean.txt')

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
    plt.xlabel('True VT')
    plt.ylabel('Predicted VT')
    plt.title(f'Scatter plot with regression line for subject {subject}')

    plt.show()

def plot_signals(true_input, uncor_input, predicted_input, i):
    # Load the signals
    true_input = os.path.join(true_input, f'{i}.txt')
    uncor_input = os.path.join(uncor_input, f'{i}.txt')
    predicted_input = os.path.join(predicted_input, f'{i}_mean.txt')

    true_signal = np.loadtxt(true_input, delimiter=',')
    uncor_signal = np.loadtxt(uncor_input, delimiter=',')
    predicted_signal = np.loadtxt(predicted_input, delimiter=',')

    # Plot the signals
    plt.figure()
    plt.plot(true_signal, label='True Signal', linewidth=3)
    plt.plot(uncor_signal, label='Uncorrected Signal', linestyle='--', linewidth=3)
    plt.plot(predicted_signal, label='Predicted Signal', linestyle='--', linewidth=3)

    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Signal Value')
    plt.title(f'Signal Comparison; subject {i}')
    plt.grid(True)
    plt.show()

    # Save the plot
    plt.savefig('signal_comparison.png')

def main():
    # Define the variables
    subject = 5

    institution = 'harvard'  # or 'kcl'
    predicted = 'IDIF'

    # Set the study name based on the institution
    if institution == 'harvard':
        study = 'ReformattedAIDataMarco'
    elif institution == 'kcl':
        study = 'ReformattedOriginalDataKCL_ALL'

    # Base paths
    base_path = os.path.join('/Users/e410377/Desktop/Ludo/AlexLudo/', study, '')

    true_input_base = os.path.join(base_path, 'patient_data/metabolite_corrected_signal_data/')
    uncor_input_base = os.path.join(base_path, 'patient_data/signal_data/')
    tac_directory_base = os.path.join(base_path, 'patient_data/time_activity_curves/')
    petframestartstop_base = os.path.join(base_path, 'patient_data/PETframestartstop/')
    workdir_base = '/Users/e410377/Desktop/AIFproj-evaluation/'
    projectdir_base = os.path.join(workdir_base, study, '')

    if institution == 'harvard':
        predicted_input_base = os.path.join(base_path, 'RESULTSNEW/metabolite_corrector_aif/test/', str(subject), '')
    elif institution == 'kcl':
        predicted_input_base = os.path.join(workdir_base, 'OUT', study, predicted, 'test/')

    # Set paths based on the value of 'institution'
    predicted_input = predicted_input_base
    true_input = true_input_base
    uncor_input = uncor_input_base
    tac_directory = tac_directory_base
    petframestartstop = petframestartstop_base
    workdir = workdir_base
    projectdir = projectdir_base

    # Display paths for verification
    print('Predicted Input Path:', predicted_input)
    print('True Input Path:', true_input)
    print('TAC Directory:', tac_directory)
    print('PET Frame Start Stop Directory:', petframestartstop)
    print('Work Directory:', workdir)
    print('Project Directory:', projectdir)

    # Extract subject numbers from filenames
    fileList = [f for f in os.listdir(true_input) if f.endswith('.txt')]
    num_subjects = len(fileList)
    subjects = [int(os.path.splitext(f)[0]) for f in fileList]

    num_values = np.loadtxt(os.path.join(true_input, '0.txt')).shape[0]  # Number of values in each file

    plot_signals(true_input, uncor_input, predicted_input, subject)

    calculatetrueVT = False
    calculateVT = True

    # Calculate true VT
    if calculatetrueVT:
        plotvt = False
        methods = ['TRUE']
        for method in methods:
            output_directory = os.path.join(workdir_base, 'OUT', study, 'VtsOUT', method)
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
                os.makedirs(os.path.join(output_directory, 'Plots'))
            run_calculate_VT(method, true_input, output_directory, petframestartstop, tac_directory, subjects, plotvt)

    methods = [predicted]
    # Calculate VT
    if calculateVT:
        plotvt = False
        #methods = [predicted]
        for method in methods:
            output_directory = os.path.join(workdir_base, 'OUT', study, 'VtsOUT', method)
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
                os.makedirs(os.path.join(output_directory, 'Plots'))
            run_calculate_VT(method, predicted_input, output_directory, petframestartstop, tac_directory, subjects, plotvt)

    # Paths for correlation plot
    path1 = os.path.join(workdir_base, 'OUT', study, 'VtsOUT', methods[0], '')
    path2 = os.path.join(workdir_base, 'OUT', study, 'VtsOUT', 'TRUE', '')

    plot_correlation_with_regression_line(subject, path1, path2)


if __name__ == "__main__":
    main()
