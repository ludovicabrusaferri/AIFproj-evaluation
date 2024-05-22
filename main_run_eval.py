import os
import numpy as np
from functions import *
np.random.seed(42)

# Set paths
dataset="ReformattedAIDataMarco"
input_path = "{0}/{1}/".format(os.path.dirname(os.getcwd()),"/Ludo/AlexLudo/")
input_path = "{0}/{1}/{2}".format(input_path, dataset, "patient_data")
workdir = "{0}/{1}/".format(os.path.dirname(os.getcwd()),"AIFproj-evaluation")
projectdir = "{0}/{1}/".format(workdir, dataset)

true_input = "{0}/{1}/".format(input_path, "/metabolite_corrected_signal_data/")
tac_directory = "{0}/{1}/".format(input_path, "/time_activity_curves/")
petframestartstop = "{0}/{1}/".format(input_path, "/PETframestartstop/")

aif_input = "{0}/{1}/{2}/{3}".format(os.path.dirname(os.getcwd()),"/Ludo/AlexLudo/", dataset, "RESULTS/")

fileList = [f for f in os.listdir(true_input) if f.endswith('.txt')]
num_subjects = len(fileList)
subjects = list(range(num_subjects))
num_values = np.loadtxt(os.path.join(true_input, '0.txt')).shape[0]

#simulateData = False  # Change to True if you want to simulate data
calculateTrueVT =False # This calculates Vt first
calculateVT = True# This calculates Vt first
do_eval = False # This runs the evaluation on calculated Vt

# Simulate Data
#if simulateData:
  #  method = 'AEIF'
  #  output_directory = "{0}/{1}/{2}/".format(projectdir, 'VtsOUT', method)
   # predicted_input = "{0}/{1}/{2}/".format(workdir, dataset, method)

   # if not os.path.exists(output_directory):
     #   os.makedirs(output_directory)
    #    os.makedirs("{0}/{1}/".format(output_directory, 'Plots'))

   # noise_level = 0.1
   # run_simulate_data(true_input, predicted_input, num_subjects, noise_level)

# Calculate VT
if calculateTrueVT:
    plot_vt = False
    methods = ['TRUE']  # Add 'TRUE' to the list of methods
    output_directory = "{0}/{1}/".format(projectdir, 'VtsOUT')

    for method in methods:
        method_output_directory = "{0}/{1}".format(output_directory, method)

        if not os.path.exists(method_output_directory):
            os.makedirs(method_output_directory)
            os.makedirs(os.path.join(method_output_directory, 'Plots'))

        predicted_input = true_input

        run_calculate_VT(method, true_input, predicted_input, method_output_directory, petframestartstop, tac_directory, subjects, plot_vt)


# Calculate VT
if calculateVT:
    plot_vt = False
    methods = ['AIF']  # Add 'TRUE' to the list of methods
    output_directory = "{0}/{1}/".format(projectdir, 'VtsOUT')

    for method in methods:
        method_output_directory = "{0}/{1}".format(output_directory, method)

        if not os.path.exists(method_output_directory):
            os.makedirs(method_output_directory)
            os.makedirs(os.path.join(method_output_directory, 'Plots'))

        predicted_input = "/Users/e410377/Desktop/Ludo/AlexLudo/ReformattedAIDataMarco/RESULTS/metabolite_corrector_aif/test/"

        run_calculate_VT(method, true_input, predicted_input, method_output_directory, petframestartstop, tac_directory, subjects, plot_vt)

# Evaluate
if do_eval:
    run_evaluation(projectdir, num_subjects, 'TRUE', 'AIF')

# Example usage
subject = 15
path1 = '/Users/e410377/Desktop/AIFproj-evaluation/ReformattedAIDataMarco/VtsOUT/AIF'
path2 = '/Users/e410377/Desktop/AIFproj-evaluation/ReformattedAIDataMarco/VtsOUT/TRUE'

plot_correlation_with_regression_line(subject, path1, path2)

print('DONE')
