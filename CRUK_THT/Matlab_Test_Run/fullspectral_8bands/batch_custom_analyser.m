
clc;clear;
close all;

% load and generate intensity and lifetime iamges
%number of files read in and the naming of output files
numberofRows = 5; %this is the number of rows OR just change to the number of samples to analyse
numberofColums = 12;

AlphaMask = 1; % set to 0 to plot with no Aplha masking applied - TO BE IMPLEMENTED
AlphaScalefactor = 2.5; % Scaling factor for alpha mask (contrast level)
oneMinusAlpha = 0; %set to 1 for a 1-Alpha plot

%scaling for Intesnity normalisation, increase if needed (if using more
%than 1 image in a single run
scalingFactorIntensity = 1.2;

%set the wavelength range to look over (need to convet drom wavelength 1 =
%500 nm, 512 = 780 nm)
firstSensorPixel = 1;
lastSensorPixel = 410;
numberOfwavelengthstoPlot = 12; % number of wavelengths to fit beteen 500 and 720nm, evenly spaced

pixeldivider = round((lastSensorPixel-firstSensorPixel)/numberOfwavelengthstoPlot);
lastSensorPixel = pixeldivider*numberOfwavelengthstoPlot;

% set number of spectral pixels for moving sum, increase if mean Tau data noisy
spectral_pixel_span = 32; 

% set threashold for when lifetime is set to NaN, based on peak bin fitted
count_threshold = 100;
mask_threshold = count_threshold;

% if you want to only want to create the lifetime data cubes:
% numberOfwavelengthstoPlot = 512,firstSensorPixel = 1 ,lastSensorPixel =
% 512 then set all the ploting / video options below to 0

plotImages = 1; % set to 1 to plot and save lifetime images set to 0 to simply save data
plotNormalisedImages = 0; % set to 1 to plot and save normalised lifetime images set to 0 to simply save data
createVideo = 1; % set to 1 to create a video of the computed lifetime images with histograms
videoQuality = 60; % set between 1 and 100 if output video too large
frameRate=20; % 45-60 works well for full spectral

%select 1st and last bin for fitting ALWAYS CHECK THESE LOOK GOOD FOR YOUR
%DATA - USE 
binToFit1 = 10;
binToFit2 = 12;

%set hitmode
histMode = 3;

%peramiters for plotting=
bin = 2; % for alpha mask
sample = 'test';

lifetimeLow = 1.5; % for stained, 0.7,  1.5 for unstained/fresh, MHorrick 1
lifetimeHigh = 2.8; % for stained, 1.7,  2.8 for unstained/fresh, MHorrricks 2

current_dir = pwd;

Wavelength = 0;

for i = firstSensorPixel:pixeldivider:lastSensorPixel
    Wavelength = Wavelength +1;
    Wave(i)  = round(i*0.5468 + 500);
end


%% batch reconstruct
% base_dir = "D:\devs\inverted";
% target_dir = "D:\devs\inverted\processed_data\flim\mats";

base_dir = "C:\Users\THT_CRUK\Documents\MATLAB\Test_Run\20221202_PutSessionNameHere";
target_dir = "C:\Users\THT_CRUK\Documents\MATLAB\Test_Output";

all_experiments = dir(base_dir);
for e = 1:size(all_experiments, 1)
    exp_name = all_experiments(e).name;
    if ~contains(exp_name, "_CR")
        continue;
    end
    
%     if endsWith(exp_name, "US")
%         continue;
%     end
    
    data_dir = strcat(base_dir, "\", exp_name, "\HistMode_no_pixel_binning");
    % reconstruct intensity and lifetime
    all_files = dir(data_dir);
    for i = 1:size(all_files, 1)
        dir_name = all_files(i).name;
        if dir_name == "." || dir_name == ".."
            continue;
        end
        disp(dir_name)

        disp(['    Loading workspace for folder: ', dir_name]);
        cd(strcat(data_dir, "\", dir_name))
        load('workspace.frame_1.mat')
        cd(current_dir)

        disp('    Producing Intensity Image')
        intensity_image = Intensity_Image_Summation(bins_array_3, frame_size_x);

        disp('    Performing Lifetime Calculations')
        % do lifetime fit calculations
        [parameters_cpu, selected_data_for_subtraction, bins_array_movsum_selected_reshaped] = test_LM_fitting_linear_gw(bins_array_3, histMode, spectral_pixel_span, binToFit1, binToFit2);
        numberofbins = size(selected_data_for_subtraction(:,1),1);
        selected_data_for_subtractionPeakbin = selected_data_for_subtraction(numberofbins,:);
        mask = selected_data_for_subtractionPeakbin;
        mask(mask<count_threshold)=0;
        mask(mask>count_threshold)=1;
        parameters_cpu(2,:) = parameters_cpu(2,:).*mask;
        tauLeastSquaresCPU = parameters_cpu(2,:); 
        tauLeastSquaresCPU(tauLeastSquaresCPU>5)=0;
        tauLeastSquaresReshaped = double(reshape(tauLeastSquaresCPU, [512 frame_size_x frame_size_x]));
        AlphaDataAll = reshape(selected_data_for_subtractionPeakbin, [512 frame_size_x frame_size_x]);

        disp(['    Save data cubes to: ', target_dir]);
        cd(target_dir);
        mkdir(strcat(exp_name, "/", dir_name));
        cd(strcat(exp_name, "/", dir_name));
        save("intensity.mat", "intensity_image");
        save("lifetime_cube.mat", "tauLeastSquaresReshaped");
        save("lifetime_alpha_cube.mat", "AlphaDataAll");
        cd(current_dir)
    end
end


%% test plot
index = 100;
alpha_scale = 2.5;
lifetime_lims = [1.0, 2.8];

for i=100:101
    intensity = squeeze(AlphaDataAll(i, :, :));
    intensity_norm = intensity/max(max(intensity));
    lifetime = squeeze(tauLeastSquaresReshaped(i, :, :));
    
    subplot(2, 2, 1)
    title(num2str(i))
    imagesc(intensity);
    colormap(jet);
    colorbar;
    
    subplot(2, 2, 2)
    imagesc(lifetime, lifetime_lims);
    colormap(jet);
    colorbar;
    
    subplot(2, 2, 3)
    imagesc(lifetime, "AlphaData", alpha_scale*intensity_norm, lifetime_lims);
    colormap(jet);
    colorbar;
    set(gca,'Color','k')
    
    subplot(2, 2, 4)
    imagesc(lifetime, "AlphaData", 1-alpha_scale*intensity_norm, lifetime_lims);
    colormap(jet);
    colorbar;
    set(gca,'Color','k')
    % lifetimeplotter(lifetime, nan, intensity_norm, 1, 2)
    
    pause(0.01);
end



