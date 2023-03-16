tic;
% prepare data
if exist('bins_array_3','var') ~= 1
    load("../workspace.frame_1.mat", "bins_array_3");
end

clearvars -except bins_array_3;

% fit model is 1 for the test of exponential decay fitting, otherwise for linear fitting
fit_model = 1;

binWidth = 0.8;

selected_bins = 3:14;

% Sum counts from contiguous spectral pixels
spectral_pixel_span = 64;

disp('Reshaping bins')
bin_16_array = bins_array_3(:,16,:,:);
bin_16_array = reshape(bin_16_array, 512, 256*256);
bin_16_array_mean = mean(bin_16_array, 2);

clear bin_16_array;

disp('Getting selected bins for background')
bins_array_selected = reshape(bins_array_3, 512, 16, 256*256);
bins_array_selected = bins_array_selected(:, selected_bins, :);

% when linear fitting, needs subtract bin 16 data as background
if fit_model ~= 1
    % Subtract background
    for j = 1:size(bins_array_selected, 3)
        for i = 1:size(bins_array_selected, 2)
            bins_array_selected(:,i,j) = bins_array_selected(:,i,j)-bin_16_array_mean;
            % Try improved mean estimate
            %bins_array_selected(:,i,j) = bins_array_selected(:,i,j)-improvedBin16Mean;
        end
    end
end

disp('Getting movmean')
array_movmean_selected = movmean(bins_array_selected,spectral_pixel_span,1);

clear bins_array_selected;

confocal_pixel_size = size(array_movmean_selected, 3);
confocal_pixel_column_size = sqrt(confocal_pixel_size);
confocal_pixel_row_size = (sqrt(confocal_pixel_size));
bin_size = size(array_movmean_selected, 2);
spectral_pixel_size = size(array_movmean_selected, 1);
%bins_array_movsum_selected_reshaped = reshape(array_movsum_selected, [bin_size, spectral_pixel_size, confocal_pixel_row_size, confocal_pixel_column_size]);
 
disp('Rearrange array')
% rearrange the array every time one column is extracted for processing
bins_array_movsum_selected_reshaped = zeros(bin_size, spectral_pixel_size, confocal_pixel_size);
for i = 1:confocal_pixel_size
    bins_array_movsum_selected_reshaped(:,:,i) = array_movmean_selected(:,:,i)';
end

clear array_movmean_selected;

selected_data_for_fitting = reshape(bins_array_movsum_selected_reshaped, ...
    size(bins_array_movsum_selected_reshaped, 1),...
    size(bins_array_movsum_selected_reshaped, 3), size(bins_array_movsum_selected_reshaped, 2));

clear bins_array_movsum_selected_reshaped;
toc;

tic;
disp('Ready for fitting')
% parameters_cpu = zeros(3, size(selected_data_for_fitting, 2)*size(selected_data_for_fitting, 3));
% states_cpu = zeros(1, size(selected_data_for_fitting, 2)*size(selected_data_for_fitting, 3));
% number_iterations_cpu = zeros(1, size(selected_data_for_fitting, 2)*size(selected_data_for_fitting, 3));
% execution_time_cpu = zeros(1, size(selected_data_for_fitting, 3));
parameters_map = containers.Map('KeyType','int32','ValueType','any');
states_map = containers.Map('KeyType','int32','ValueType','any');
number_iterations_map = containers.Map('KeyType','int32','ValueType','any');
execution_time_map = containers.Map('KeyType','int32','ValueType','any');
i=10;
%parfor i = 1:size(selected_data_for_fitting, 3)
    index = i*size(selected_data_for_fitting, 2);
    [param, states, num_iter, time] = LM_fitting(selected_data_for_fitting(:, :, i), binWidth, ModelID.EXPONENTIAL_3_PARAMS, 0);
    parameters_map(i) = param;
    states_map(i) = states;
    number_iterations_map(i) = num_iter;
    execution_time_map(i) = time;
%end
toc;
