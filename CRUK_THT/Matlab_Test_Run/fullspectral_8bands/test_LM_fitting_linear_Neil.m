%% prepare data
if exist('bins_array_3','var') ~= 1
    load("../workspace.frame_1.mat", "bins_array_3");
end

clearvars -except bins_array_3;

% fit model is 1 for the test of exponential decay fitting, otherwise for linear fitting
fit_model = 1;

binWidth = 0.8;
selected_bins = 8:13;
%timeBins = [0.8 1.6 2.4 3.2 4.0 4.8 5.6]';
timeBins = linspace(0, size(selected_bins, 2) - 1, size(selected_bins, 2))*binWidth;
timeBins2 = [ones(length(timeBins),1) timeBins'];

% Sum counts from contiguous spectral pixels
spectral_pixel_span = 64;

disp('Reshaping bins');
bin_16_array = bins_array_3(:,16,:,:);
bin_16_array = reshape(bin_16_array, 512, 256*256);
bin_16_array_mean = mean(bin_16_array, 2);

clear bin_16_array;

disp('Getting selected bins for background');
bins_array_selected = reshape(bins_array_3, 512, 16, 256*256);
bins_array_selected = bins_array_selected(:, selected_bins, :);

% Subtract background
for j = 1:size(bins_array_selected, 3)
    for i = 1:size(bins_array_selected, 2)
        bins_array_selected(:,i,j) = bins_array_selected(:,i,j)-bin_16_array_mean;
        % Try improved mean estimate
        %bins_array_selected(:,i,j) = bins_array_selected(:,i,j)-improvedBin16Mean;
    end
end

disp('Getting movmean');
array_movmean_selected = movmean(bins_array_selected,spectral_pixel_span,1);

clear bins_array_selected;

confocal_pixel_size = size(array_movmean_selected, 3);
confocal_pixel_column_size = sqrt(confocal_pixel_size);
confocal_pixel_row_size = (sqrt(confocal_pixel_size));
bin_size = size(array_movmean_selected, 2);
spectral_pixel_size = size(array_movmean_selected, 1);
%bins_array_movsum_selected_reshaped = reshape(array_movsum_selected, [bin_size, spectral_pixel_size, confocal_pixel_row_size, confocal_pixel_column_size]);
 
disp('Rearrange array');
% rearrange the array every time one column is extracted for processing
bins_array_movsum_selected_reshaped = zeros(bin_size, spectral_pixel_size, confocal_pixel_size);
for i = 1:confocal_pixel_size
    bins_array_movsum_selected_reshaped(:,:,i) = array_movmean_selected(:,:,i)';
end

clear array_movmean_selected;

%% start fitting
selected_data_for_fitting = reshape(bins_array_movsum_selected_reshaped, ...
    size(bins_array_movsum_selected_reshaped, 1),...
    size(bins_array_movsum_selected_reshaped, 2)*size(bins_array_movsum_selected_reshaped, 3));

%clear bins_array_movsum_selected_reshaped;

% spectral_pixel =200;
% selected_data_for_fitting = bins_array_movsum_selected_reshaped(:,spectral_pixel, :);
% selected_data_for_fitting = reshape(selected_data_for_fitting, size(bins_array_movsum_selected_reshaped, 1), ...
%     size(bins_array_movsum_selected_reshaped, 3));
%selected_data_for_fitting(selected_data_for_fitting < 10) = 0.01;
selected_data_for_fitting = log(selected_data_for_fitting);
selected_data_for_fitting = real(selected_data_for_fitting);

% disp('Gpufit linear 1D fitting');
% [parameters_cpu, states_cpu, number_iterations_cpu, execution_time_cpu] = ...
%     LM_fitting(selected_data_for_fitting, binWidth, ModelID.LINEAR_1D, 0);
% parameters_cpu(2,:) = 1 ./ parameters_cpu(2,:);
% disp("execution_time_cpu: " + execution_time_cpu + " seconds");

% [parameters_gpu, states_gpu, number_iterations_gpu, execution_time_gpu] = ...
%     LM_fitting(selected_data_for_fitting, binWidth, ModelID.LINEAR_1D, 1);
% parameters_gpu(2,:) = 1 ./ parameters_gpu(2,:);
% disp("execution_time_gpu: " + execution_time_gpu + " seconds");
% 
tic;
disp('Least squares lifetime fitting');
parameters_old_cpu = zeros(2, size(selected_data_for_fitting, 2));
for i = 1: size(selected_data_for_fitting, 2)
    selectedBinsForWavelengthAndConfocalPixel = selected_data_for_fitting(:, i);
    fit = timeBins2\selectedBinsForWavelengthAndConfocalPixel;
    parameters_old_cpu(:,i) = [fit(1) real(1/fit(2))];
end
execution_time_old_cpu = toc;
disp("execution_time_old_cpu: " + execution_time_old_cpu + " seconds");

% tic;
% binArrays_subset = reshape(bins_array_3, size(bins_array_3, 1), size(bins_array_3, 2), ...
%     size(bins_array_3, 3)*size(bins_array_3, 4));
% parameters_old_gpu = lifetime_leastsquare_gpu(binArrays_subset);
% execution_time_old_gpu = toc;
% disp("execution_time_old_gpu: " + execution_time_old_gpu + " seconds");