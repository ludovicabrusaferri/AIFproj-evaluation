% Set paths
clear all
rng(42);
path = '/Users/e410377/Desktop/Ludo/AlexLudo/';
%project='/ReformattedOriginalDataKCL_ALL';
project='/ReformattedAIDataMarco/';
%addpath(genpath('/Users/e410377/Desktop/PETAnalysisPaper/utility'));
workdir = path;
script_directory = '/Users/e410377/Desktop/AIFproj-evaluation/';


true_input = [path, project, '/patient_data/metabolite_corrected_signal_data'];
tac_directory = [path ,project, '/patient_data/time_activity_curves/'];
petframestartstop = [path, project, '/patient_data/PETframestartstop'];
aif_input = [path, project, '/RESULTS/metabolite_corrector_aif/test/'];
fileList = dir(fullfile(true_input, '*.txt'));
% Determine the number of subjects based on the number of files
num_subjects = numel(fileList); 
subjects = 0:num_subjects-1; % Make a list from 0 to 51
num_values = size(load(fullfile(true_input, '0.txt')),1); % Number of values in each file


calculatetrueVT = true;
calculateVt=true;

subject = 15;




%% DO VT

if calculatetrueVT
    plotvt=false;
    method = {'TRUE'};
    for k = method
        current_method = k{1};
        output_directory = [script_directory ,'OUT',project,'/VtsOUT/', current_method];
  
        if ~exist(output_directory, 'dir')
            mkdir(output_directory);
            mkdir([output_directory, '/Plots'])
        end
        run_calculate_VT(current_method,true_input,output_directory, petframestartstop,tac_directory,subjects, plotvt)
    end
end

if calculateVt
    plotvt=false;
    method = {'AIF'};
    for k = method
        current_method = k{1};
        output_directory = [script_directory ,'OUT',project,'/VtsOUT/', current_method];
  
        if ~exist(output_directory, 'dir')
            mkdir(output_directory);
            mkdir([output_directory, '/Plots'])
        end
        run_calculate_VT(current_method,aif_input,output_directory, petframestartstop,tac_directory,subjects, plotvt)
    end
end
%%

path1 = [script_directory 'OUT', project, '/VtsOUT/' method{1}];
path2 = [script_directory 'OUT', project, '/VtsOUT/TRUE'];

plot_correlation_with_regression_line(subject, path1, path2);



%% FUNCTIONS
function plot_correlation_with_regression_line(subject, path1, path2)
    % Function to read a single float value from each file
    function data = read_data_files(path, subject)
        files = dir(fullfile(path, sprintf('%d_*.txt', subject)));
        data = [];
        for i = 1:length(files)
            file_path = fullfile(path, files(i).name);
            value = load(file_path);
            data = [data; value];
        end
    end

    % Read data from both directories
    data1 = read_data_files(path1, subject);
    data2 = read_data_files(path2, subject);

    % Check if we have at least one value from each directory
    if isempty(data1) || isempty(data2)
        error('Not enough data files found to plot correlation.');
    end

    % Plotting scatter plot
    figure;
    scatter(data1, data2, 'filled');
    hold on;

    % Adding regression line
    p = polyfit(data1, data2, 1);
    y_pred = polyval(p, data1);
    plot(data1, y_pred, 'r-', 'LineWidth', 2);

    % Plotting identity line
    min_val = min(min(data1), min(data2));
    max_val = max(max(data1), max(data2));
    plot([min_val, max_val], [min_val, max_val], 'k--', 'LineWidth', 2);

    % Equation of the regression line
    equation = sprintf('y = %.2fx + %.2f', p(1), p(2));

    % Adding equation to the plot
    text(0.1, 0.9, equation, 'Units', 'normalized', 'FontSize', 12);

    % Adding labels and title
    xlabel('Values from first data set');
    ylabel('Values from second data set');
    title(sprintf('Scatter plot with regression line for subject %d', subject));

    hold off;
end
%% DO EVAL

function run_calculate_VT(method,true_input,output_directory, petframestartstop,tac_directory,subjects,plotvt)
    for i = subjects
        disp(['============ SUBJ ', num2str(i), ' and i: ', num2str(i), ' \n\n =======\n']);
        tstar=15;
        frame_time_fn = fullfile(petframestartstop, [num2str(i), '.txt']);
        if strcmp(method, 'AIF')
            plasma_Lin_filename = fullfile(true_input, num2str(i),[num2str(i), '_mean.txt']);
        elseif strcmp(method, 'TRUE')
            plasma_Lin_filename = fullfile(true_input, [num2str(i), '.txt']);
        end
    
         % Check if the plasma_Lin_filename exists
        if ~exist(plasma_Lin_filename, 'file')
            disp(['File does not exist: ', plasma_Lin_filename, '. Skipping subject ', num2str(i)]);
            continue;
        end
        

        cd(tac_directory);
        cd(fullfile(tac_directory, num2str(i)));
        tacs = dir([num2str(i), '_*.txt']);
    
        for j = 1:length(tacs)
            target_fn = tacs(j).name;
            [~, tac_name, ~] = fileparts(target_fn);
            reference_fn = plasma_Lin_filename;
    
            output_fn = fullfile(output_directory, [tac_name, '_Logan_Vt_', num2str(tstar)]);
    
            if ~exist([output_fn, '.!txt'], 'file')
                vt=calculate_logan_vt(frame_time_fn,reference_fn, target_fn, tstar, output_fn, output_directory, reference_fn,plotvt);
                
                fprintf('vt=%f\n', vt);
                if vt<0
                    error('vt cannot be negative.');
                end
            end
        end
        
        cd(tac_directory);
    end
end



function Vt = calculate_logan_vt(frame_time_filename, reference_filename, target_filename, tstar, output_filename, output_directory, reference_name, plotvt)
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
    
    %% AVOIDINIG DIVISION BY ZERO
    for i = 1:2
        if X_matrix(i) > 150 || Y_vector(i) > 150
            fprintf("I AM HERE")
            X_matrix(i) = NaN;
            Y_vector(i) = NaN;
        end
    end

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

    % Check for NaN in Vt
    if isnan(Vt)
        error('Vt is NaN. Check input values!');
    end

    % Plot figures if specified
    if plotvt
        disp('Making the figures now..');
        h = figure('Visible', 'on');
        plot(X_matrix(:, 1), Y_vector, '*', X_selected(:, 1), X_selected * regression_coefficients, 'k');
        hold on;
        plot(X_matrix(tstar_index, 1), Y_vector(tstar_index), 'o', 'MarkerSize', 10);
        title(['Logan Vt: target-', reference_name, ', reference-plasma, tstar: ', num2str(tstar), ' minutes']);
        ylabel('Y'); xlabel('X'); set(gcf, 'Color', 'w');
        % Set equal scaling for both axes
        axis equal;
    
        grid on;
        saveas(h, fullfile(output_directory, 'Plots', ['_Logan_Vt_', num2str(tstar), '.png']), 'png');
        close(h);
        disp('Figures DONE..');
    end

    % Save Vt to a text file
    fid = fopen([output_filename, '.txt'], 'w');
    fprintf(fid, '%f\n', Vt);
    fclose(fid);

end


