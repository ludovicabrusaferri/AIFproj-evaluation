clear all;

% Define the institution and predicted variable
institution = 'harvard';  % 'harvard' or 'kcl'
predicted = 'AIF';

% Set the study name based on the institution
switch institution
    case 'harvard'
        study = 'ReformattedAIDataMarco';
    case 'kcl'
        study = 'ReformattedOriginalDataKCL_ALL';
end

% Base paths
base_path = fullfile('/Users/e410377/Desktop/Ludo/AlexLudo/', study, '/');
true_input_base = fullfile(base_path, 'patient_data/metabolite_corrected_signal_data/');
tac_directory_base = fullfile(base_path, 'patient_data/time_activity_curves/');
petframestartstop_base = fullfile(base_path, 'patient_data/PETframestartstop/');
workdir_base = '/Users/e410377/Desktop/AIFproj-evaluation/';
projectdir_base = fullfile(workdir_base, study, '/');

% Predicted input path based on the institution
switch institution
    case 'harvard'
        predicted_input_base = fullfile(base_path, 'RESULTSNEW/metabolite_corrector_aif/test/', predicted, '/');
    case 'kcl'
        predicted_input_base = fullfile(workdir_base, 'OUT', study, predicted, '/test/');
end

% Find subjects in the predicted input base directory
files = dir(fullfile(predicted_input_base, '*.txt'));
subjects = unique(arrayfun(@(x) strtok(x.name, '_'), files, 'UniformOutput', false));
subjectlist = cellfun(@str2double, subjects);

disp('Subject List:');
disp(subjectlist);

% Initialize storage for all VT values
all_true_vt_values = [];
all_predicted_vt_values = [];
all_true_signal = [];
all_pred_signal = [];


% Loop over each subject
for subj_idx = 1:length(subjectlist)
    subject = subjects{subj_idx};
    
    % Set paths
    true_input = true_input_base;
    predicted_input = predicted_input_base;
    tac_directory = tac_directory_base;
    petframestartstop = petframestartstop_base;
    workdir = workdir_base;
    projectdir = projectdir_base;

    % Load the true and predicted input for the given subject
    true_input_filename = fullfile(true_input, [num2str(subject), '.txt']);
    predicted_input_filename = fullfile(predicted_input, [num2str(subject), '_mean.txt']);
    
    true_signal = load(true_input_filename);
    predicted_signal = load(predicted_input_filename);

    % Calculate VT for all TACs for the given subject using both true and predicted blood data
    tstar = 15; % You can adjust this as needed

    % Define paths for TACs and frame times
    tac_path = fullfile(tac_directory, num2str(subject));
    frame_time_fn = fullfile(petframestartstop, [num2str(subject), '.txt']);

    % Load all TAC files for the subject
    tac_files = dir(fullfile(tac_path, [num2str(subject), '_*.txt']));
    num_tacs = length(tac_files);

    true_vt_values = zeros(num_tacs, 1);
    predicted_vt_values = zeros(num_tacs, 1);

    for i = 1:num_tacs
        tac_filename = tac_files(i).name;
        target_fn = fullfile(tac_path, tac_filename);
        output_true_fn = fullfile(workdir_base, 'OUT', study, 'VtsOUT', 'TRUE', [tac_filename, '_Logan_Vt_', num2str(tstar)]);
        output_pred_fn = fullfile(workdir_base, 'OUT', study, 'VtsOUT', predicted, [tac_filename, '_Logan_Vt_', num2str(tstar)]);

        % Create directories if they do not exist
        output_true_dir = fileparts(output_true_fn);
        output_pred_dir = fileparts(output_pred_fn);

        if ~exist(output_true_dir, 'dir')
            mkdir(output_true_dir);
        end

        if ~exist(output_pred_dir, 'dir')
            mkdir(output_pred_dir);
        end

        % Calculate VT using true blood data
        true_vt = calculate_logan_vt(frame_time_fn, true_input_filename, target_fn, tstar, output_true_fn, workdir_base, false);
        true_vt_values(i) = true_vt;

        % Calculate VT using predicted blood data
        predicted_vt = calculate_logan_vt(frame_time_fn, predicted_input_filename, target_fn, tstar, output_pred_fn, workdir_base, false);
        predicted_vt_values(i) = predicted_vt;

        fprintf('Subject %s, TAC %d: True VT = %f, Predicted VT = %f\n', subject, i, true_vt, predicted_vt);
    end

    % Store all VT values
    all_true_vt_values = [all_true_vt_values, true_vt_values];
    all_predicted_vt_values = [all_predicted_vt_values, predicted_vt_values];
    all_true_signal = [all_true_signal,true_signal];
    all_pred_signal = [all_pred_signal,predicted_signal];
end

%%
% Plot settings

num_subjects = numel(subjectlist);
 figure(1)

for i = 1:num_subjects
   
    subplot(2,round(num_subjects/2),i)
    subject = subjectlist(i);
    
    true_signal = all_true_signal(:,i);
    predicted_signal =  all_pred_signal(:,i);
    

    plot(true_signal(15:end), 'DisplayName', 'True Signal', 'LineWidth', 2);
    hold on;
    plot(predicted_signal(15:end), 'DisplayName', 'Predicted Signal', 'LineWidth', 2);
    hold off;
    legend('show');
    title([predicted,':Subj. ', num2str(subject)]);
    xlabel('Time');
    ylabel('Uptake');
    grid on;
    set(gca, 'FontSize', 25);
    set(gcf, 'Color', 'w');
    x_range = 15:length(true_signal); % Assuming true_signal and predicted_signal have the same length
    xticks(1:length(x_range));
    xticklabels(x_range);
    
    
    hold off
end


    figure(2)
for i = 1:num_subjects
    subplot(2,round(num_subjects/2),i)   
    subject = subjectlist(i);
    true_vt_values = all_true_vt_values(:,i);
    predicted_vt_values = all_predicted_vt_values(:,i);
    scatter(true_vt_values, predicted_vt_values, 'filled');
    hold on;
    
    % Adding regression line
    p = polyfit(true_vt_values, predicted_vt_values, 1);
    y_pred = polyval(p, true_vt_values);
    plot(true_vt_values, y_pred, 'r-', 'LineWidth', 2);
    
    % Plotting identity line
    min_val = min([true_vt_values; predicted_vt_values]);
    max_val = max([true_vt_values; predicted_vt_values]);
    plot([min_val, max_val], [min_val, max_val], 'k--', 'LineWidth', 2);
    
    % Equation of the regression line
    equation = sprintf('y = %.2fx + %.2f', p(1), p(2));
    
    % Adding equation to the plot
    text(0.1, 0.9, equation, 'Units', 'normalized', 'FontSize', 12);
    
    % Adding labels and title
    xlabel('True VT');
    ylabel('Predicted VT');
    title([predicted,': Vt ', num2str(subject)]);
    set(gca, 'FontSize', 20);
    set(gcf, 'Color', 'w');
    hold off;
    set(gcf, 'Color', 'w');

end

% Function to calculate Logan VT
function Vt = calculate_logan_vt(frame_time_filename, reference_filename, target_filename, tstar, output_filename, output_directory, plotvt)
    % Load data
    time_all = load(frame_time_filename);
    reference_tac = load(reference_filename);
    target_tac = load(target_filename);

    % Transpose if target is a row vector
    if size(target_tac, 1) == 1
        target_tac = target_tac';
    end

    % Time calculations
    time = (time_all(:, 2) / 60 + time_all(:, 1) / 60) / 2;
    dt = time_all(:, 2) / 60 - time_all(:, 1) / 60;

    % Calculate integrated reference and target
    int_ref = cumtrapz(time, reference_tac);
    int_target = cumtrapz(time, target_tac);
    intercept = ones(size(reference_tac));

    % Construct matrices for linear regression
    X_matrix = [int_ref ./ target_tac, intercept];
    Y_vector = int_target ./ target_tac;

    % Select data after tstar
    tstar_index = find(time >= tstar, 1, 'first');
    X_selected = X_matrix(tstar_index:end, :);
    Y_selected = Y_vector(tstar_index:end);

    % Weight matrix
    weight_matrix = diag(ones(size(dt)));
    weight_matrix_selected = weight_matrix(tstar_index:end, tstar_index:end);

    % Perform linear regression
    regression_coefficients = (weight_matrix_selected * X_selected) \ (weight_matrix_selected * Y_selected);
    Vt = regression_coefficients(1);
    
    if isnan(Vt)
        error('The calculated Vt is NaN. Please check the input data and calculations.');
    end

    % Plot figures if specified
    if plotvt
        figure;
        plot(X_matrix(:, 1), Y_vector, '*', X_selected(:, 1), X_selected * regression_coefficients, 'k');
        hold on;
        plot(X_matrix(tstar_index, 1), Y_vector(tstar_index), 'o', 'MarkerSize', 10);
        title(['Logan Vt: target-', reference_name, ', reference-plasma, tstar: ', num2str(tstar), ' minutes']);
        ylabel('Y'); xlabel('X'); set(gcf, 'Color', 'w');
        grid on;
        saveas(gcf, fullfile(output_directory, 'Plots', ['_Logan_Vt_', num2str(tstar), '.png']), 'png');
        close(gcf);
    end

    % Save Vt to a text file
    fid = fopen([output_filename, '.txt'], 'w');
    fprintf(fid, '%f\n', Vt);
    fclose(fid);
end


