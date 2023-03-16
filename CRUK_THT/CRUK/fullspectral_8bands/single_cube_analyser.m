%% plot decay
clear all;
n_bins = 16;
n_spectrum = 8;
frame_size = 256;

if exist('bins_array_3','var') == 0
    base_dir = "z:\Devs\data\CRUK_EDD\20221206_8_channels\HistMode_full_8bands_pixel_binning_inFW\";
    data_dir = "test\";
    data_file = "workspace.frame_1.mat";
    load(base_dir + data_dir + data_file, "bins_array_3");
end
spectrum = 1;
x_index = 150;
y_index = 96;
decay_bin = bins_array_3(x_index, y_index, :, spectrum);
decay_bin = reshape(decay_bin, n_bins, 1);
plot([1:n_bins]', decay_bin);


%% load and generate intensity and lifetime iamges
%number of files read in and the naming of output files
numberofRows = 1; %this is the number of rows OR just change to the number of samples to analyse
numberofColums = 1;

AlphaMask = 1; % set to 0 to plot with no Aplha masking applied - TO BE IMPLEMENTED
AlphaScalefactor = 2.5; % Scaling factor for alpha mask (contrast level)
oneMinusAlpha = 0; %set to 1 for a 1-Alpha plot

%scaling for Intesnity normalisation, increase if needed (if using more
%than 1 image in a single run
scalingFactorIntensity = 1.2;

%set the wavelength range to look over (need to convet drom wavelength 1 =
%500 nm, 512 = 780 nm)
firstSensorPixel = 1;
lastSensorPixel = 8;
numberOfwavelengthstoPlot = 8; % number of wavelengths to fit beteen 500 and 720nm, evenly spaced

pixeldivider = round((lastSensorPixel-firstSensorPixel)/numberOfwavelengthstoPlot);
lastSensorPixel = pixeldivider*numberOfwavelengthstoPlot;

% set number of spectral pixels for moving sum, increase if mean Tau data noisy
spectral_pixel_span = 1;

% set threashold for when lifetime is set to NaN, based on peak bin fitted
count_threshold = 100;
mask_threshold = count_threshold;

% if you want to only want to create the lifetime data cubes:
% numberOfwavelengthstoPlot = 512,firstSensorPixel = 1 ,lastSensorPixel =
% 512 then set all the ploting / video options below to 0

plotImages = 0; % set to 1 to plot and save lifetime images set to 0 to simply save data
plotNormalisedImages = 0; % set to 1 to plot and save normalised lifetime images set to 0 to simply save data
createVideo = 1; % set to 1 to create a video of the computed lifetime images with histograms
videoQuality = 60; % set between 1 and 100 if output video too large
frameRate=20; % 45-60 works well for full spectral

%select 1st and last bin for fitting ALWAYS CHECK THESE LOOK GOOD FOR YOUR
%DATA - USE
binToFit1 = 10;
binToFit2 = 14;

%set hitmode
histMode = 3;

%peramiters for plotting=
bin = 2; % for alpha mask
sample = 'test';

lifetimeLow = 1.5; % for stained, 0.7,  1.5 for unstained/fresh, MHorrick 1
lifetimeHigh = 3.5; % for stained, 1.7,  2.8 for unstained/fresh, MHorrricks 2



current_dir = pwd;

Wavelength = 0;
wl_lower = 470; % nm
wl_upper = 600; % nm
wl_step = (wl_upper - wl_lower)/lastSensorPixel;
for i = firstSensorPixel:pixeldivider:lastSensorPixel
    Wavelength = Wavelength +1;
    Wave(i)  = round(i*wl_step + wl_lower);
end


% path
data_file = 'z:\Devs\data\CRUK_EDD\20221206_8_channels\HistMode_full_8bands_pixel_binning_inFW\test\workspace.frame_1.mat';
target_dir = "./processed/";
addpath(target_dir);
addpath("../fullspectral_8bands/");

if ~exist(target_dir, "dir")
    mkdir(target_dir);
end

% reconstruct intensity and lifetime, and save to mat files per experiment
disp(['    Loading workspace for folder: ', data_file]);
load(data_file);

% the code is for channels*bins*frame_size*frame_size, whereas the new
% array is in frame_size*frame_size*bins*channels
bins_array_3 = permute(bins_array_3, [4 3 1 2]);

disp('    Producing Intensity Image')
% intensity_image = Intensity_Image_Summation(bins_array_3, frame_size);
intensity_image = sum(sum(bins_array_3, 1), 2);

disp('    Performing Lifetime Calculations')
% do lifetime fit calculations
[parameters_cpu, selected_data_for_subtraction, bins_array_movsum_selected_reshaped] = test_LM_fitting_linear_gw(bins_array_3, histMode, spectral_pixel_span, binToFit1, binToFit2, frame_size, n_bins, n_spectrum);
numberofbins = size(selected_data_for_subtraction(:,1),1);
selected_data_for_subtractionPeakbin = selected_data_for_subtraction(numberofbins,:);
mask = selected_data_for_subtractionPeakbin;
mask(mask<count_threshold)=0;
mask(mask>count_threshold)=1;
parameters_cpu(2,:) = parameters_cpu(2,:).*mask;
tauLeastSquaresCPU = parameters_cpu(2,:);
tauLeastSquaresCPU(tauLeastSquaresCPU>5)=0;
tauLeastSquaresReshaped = reshape(tauLeastSquaresCPU, [n_spectrum frame_size_x frame_size_x]);
AlphaDataAll = reshape(selected_data_for_subtractionPeakbin, [n_spectrum frame_size_x frame_size_x]);

%% plot to target dir
save_dir = "./processed";
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

overallNormalisationValue = max(intensity_image);
overallNormalisationValue = overallNormalisationValue/scalingFactorIntensity;

disp("    plot lifetime image");
a = 0;
for z = firstSensorPixel:pixeldivider:lastSensorPixel
    a = a + 1;
    Currentwavelength = round(z*wl_step + wl_lower);

    OverallNormalisedAlphaFactor = max(max(max(AlphaDataAll)));

    ImageData = tauLeastSquaresReshaped;
    AlphaData = AlphaDataAll;

    ImagetoPlot = squeeze(ImageData(z, : , :));

    % intensity
    folder = strcat(save_dir, "/intensity_", num2str(Currentwavelength),'nm.jpg');
    intensityplotter(squeeze(AlphaData(z,:,:))/OverallNormalisedAlphaFactor, folder)

    % lifetime only
    folder = strcat(save_dir, "/lifetime_", num2str(Currentwavelength),'nm.jpg');
    lifetimeplotter(ImagetoPlot, folder, 1, lifetimeLow, lifetimeHigh)

    % lifetime with alpha
    AlphaData = squeeze(AlphaData(z,:,:));
    AlphatoPlot = AlphaScalefactor*squeeze(AlphaData)/OverallNormalisedAlphaFactor;
    folder = strcat(save_dir, "/lifetime_alpha_", num2str(Currentwavelength),'nm.jpg');
    lifetimeplotter(ImagetoPlot, folder, AlphatoPlot, lifetimeLow, lifetimeHigh)

    % alpha-weighted lifetime
    AlphatoPlot = AlphaScalefactor*squeeze(AlphaData)/OverallNormalisedAlphaFactor;
    %folder = strcat(save_dir, "/", allXY{i}, "/images/lifetime_weighted_", num2str(Currentwavelength),'nm.jpg');
    folder = strcat(save_dir, "/lifetime_weighted_", num2str(Currentwavelength),'nm.jpg');
    %         ImagetoPlotScaled = ImagetoPlot.*AlphatoPlot;
    ImagetoPlotScaled = lifetime_blending(ImagetoPlot, AlphatoPlot);
    lifetimeplotter(ImagetoPlotScaled, folder, 1, 0.05, lifetimeHigh)

    % lifetime with 1-alpha
    AlphatoPlot = AlphaScalefactor*(1-(AlphaData/OverallNormalisedAlphaFactor));
    folder = strcat(save_dir, "/lifetime_alpha_inv_", num2str(Currentwavelength),'nm.jpg');
    lifetimeplotter(ImagetoPlot, folder, AlphatoPlot, lifetimeLow, lifetimeHigh);
end
