import os
import numpy as np
from functions import *
np.random.seed(42)

# Define the variable
subject = 14
bool = 'kcl'  # or 'kcl'

# Set the study name based on the institution
if bool == 'harvard':
    study = 'ReformattedAIDataMarco'
elif bool == 'kcl':
    study = 'ReformattedOriginalDataKCL_ALL'

# Base paths
base_path = f"/Users/e410377/Desktop/Ludo/AlexLudo/{study}/"

true_input_base = f"{base_path}patient_data/metabolite_corrected_signal_data/"
tac_directory_base = f"{base_path}patient_data/time_activity_curves/"
petframestartstop_base = f"{base_path}patient_data/PETframestartstop/"
workdir_base = "/Users/e410377/Desktop/AIFproj-evaluation/"
projectdir_base = f"{workdir_base}{study}/"

if bool == 'harvard':
    predicted_input_base = f"{base_path}RESULTSNEW/metabolite_corrector_aif/test/{subject}/"
elif bool == 'kcl':
    predicted_input_base = f"{workdir_base}/OUT/{study}/AIF/test/"

# Set paths based on the value of 'bool'
predicted_input = predicted_input_base
true_input = true_input_base
tac_directory = tac_directory_base
petframestartstop = petframestartstop_base
workdir = workdir_base
projectdir = projectdir_base

# Continue with the rest of your code

# Paths for KCL would be similarly defined here

fileList = [f for f in os.listdir(true_input) if f.endswith('.txt')]
num_subjects = len(fileList)
subjects = list(range(num_subjects))
num_values = np.loadtxt(os.path.join(true_input, '0.txt')).shape[0]

# simulateData = False  # Change to True if you want to simulate data
calculateTrueVT = True  # This calculates Vt first
calculateVT = True  # This calculates Vt first

# Calculate VT
if calculateTrueVT:
    plot_vt = False
    methods = ['TRUE']  # Add 'TRUE' to the list of methods
    output_directory = f"{workdir}OUT/{study}/VtsOUT"

    for method in methods:
        method_output_directory = f"{output_directory}/{method}"

        if not os.path.exists(method_output_directory):
            os.makedirs(method_output_directory)
            os.makedirs(os.path.join(method_output_directory, 'Plots'))

        input = true_input

        run_calculate_VT(method, input, method_output_directory, petframestartstop, tac_directory, subjects, plot_vt)

# Calculate VT
if calculateVT:
    plot_vt = False
    methods = ['AIF']  # Add 'AIF' to the list of methods
    output_directory = f"{workdir}OUT/{study}/VtsOUT"

    for method in methods:
        method_output_directory = f"{output_directory}/{method}"

        if not os.path.exists(method_output_directory):
            os.makedirs(method_output_directory)
            os.makedirs(os.path.join(method_output_directory, 'Plots'))

        input = predicted_input

        run_calculate_VT(method, input, method_output_directory, petframestartstop, tac_directory, subjects, plot_vt)

path1 = f"{workdir}OUT/{study}/VtsOUT/AIF"
path2 = f"{workdir}OUT/{study}/VtsOUT/TRUE"
print(path1)
plot_correlation_with_regression_line(subject, path1, path2)

print('DONE')
