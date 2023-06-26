function [parameters, states, number_iterations, execution_time] = LM_fitting(array, binWidth, model_id, onGPU)
%LM_FITTING Summary of this function goes here
%   Detailed explanation goes here

assert(length(size(array)) == 2, ...
    "data needs to be 2D array, first array is the number of points for fitting");

if ~(isa(array, "single"))
    data = single(array);
end

number_points = size(data, 1);
number_fits = size(data, 2);

% exponential decay: y = a*exp(-b*x)+c
number_parameter = 0;

% model ID
if isempty(model_id)
    model_id = ModelID.LINEAR_1D;
    number_parameter = 2;
elseif model_id == ModelID.LINEAR_1D
    number_parameter = 2;
elseif model_id == ModelID.EXPONENTIAL_3_PARAMS
    number_parameter = 3;
else
    disp("Unknown modelID");
    disp("modelID needs to be either ModelID.LINEAR_1D or ModelID.EXPONENTIAL_3_PARAMS");
    return;
end

% initial parameters
initial_parameters = ones(number_parameter, number_fits, 'single');

% tolerance
tolerance = 1e-5;

% maximum number of iterations
max_n_iterations = 10000;

% estimator id
estimator_id = EstimatorID.LSE;

% user_info, used as X axis during fitting
user_info = single(linspace(0, number_points - 1, number_points)*binWidth);

if isempty(onGPU) || onGPU == 0  % use cpu
    [parameters, states, ~, number_iterations, execution_time] = ...
        cpufit(data, [], model_id, initial_parameters, tolerance, max_n_iterations, [], estimator_id, user_info);
else  % use GPU
    assert(gpufit_cuda_available(), "No suitable GPU");
    [parameters, states, ~, number_iterations, execution_time] = ...
        gpufit(data, [], model_id, initial_parameters, tolerance, max_n_iterations, [], estimator_id, user_info);
end

if number_parameter == 3
    parameters(2,:) = 1./parameters(2,:);
end

end