clear all;

% load and generate intensity and lifetime iamges
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

plotImages = 0; % set to 1 to plot and save lifetime images set to 0 to simply save data
plotNormalisedImages = 0; % set to 1 to plot and save normalised lifetime images set to 0 to simply save data
createVideo = 1; % set to 1 to create a video of the computed lifetime images with histograms
videoQuality = 60; % set between 1 and 100 if output video too large
frameRate=20; % 45-60 works well for full spectral

%select 1st and last bin for fitting ALWAYS CHECK THESE LOOK GOOD FOR YOUR
%DATA - USE 
binToFit1 = 11;
binToFit2 = 14;

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


% path
data_dir = "G:\inverted\20210409_CR71A_4_FOV160_testing\HistMode_no_pixel_binning";
target_dir = "G:\inverted\processed\20210409_CR71A_4_FOV160_testing";
addpath(target_dir);
addpath("F:\Backup\UoE\D_driver\Dev\Matlab\fullspectral_22_07_2019");

%% reconstruct intensity and lifetime, and save to mat files per experiment
if ~exist(target_dir, "dir")
    mkdir(target_dir);
end
    
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
    mkdir(dir_name);
    cd(dir_name);
    save("intensity.mat", "intensity_image", "-v7.3");
    save("lifetime_cube.mat", "tauLeastSquaresReshaped", "-v7.3");
    save("lifetime_alpha_cube.mat", "AlphaDataAll", "-v7.3");
    cd(current_dir)
end


%% load all reconstructed data per experiment, and save them to single mat files
%clear all;
addpath(target_dir);
all_files = dir(target_dir);
allXY = {};
allIntensityImages={};
lifetimeImageData={};
lifetimeAlphaData={};
a = 0;
for i = 1:size(all_files, 1)
    dir_name = all_files(i).name;
    if dir_name == "." || dir_name == ".." || isfile(dir_name)
        continue;
    end
    
    a = a + 1;
    allXY{a, 1} = dir_name;
    
    cd(strcat(target_dir, "/", dir_name));
    
    clear intensity_image
    load("intensity.mat");
    allIntensityImages{a, 1} = intensity_image;
    
    clear tauLeastSquaresReshaped
    load("lifetime_cube.mat");
    lifetimeImageData{a, 1} = single(tauLeastSquaresReshaped);
    
    clear AlphaDataAll
    load("lifetime_alpha_cube.mat");
    lifetimeAlphaData{a, 1} = single(AlphaDataAll);
    
end

cd(target_dir);
save("allXY.mat", "allXY");
save("allIntensityImages.mat", "allIntensityImages");
save("lifetimeImageData.mat", "lifetimeImageData", "-v7.3");
save("lifetimeAlphaData.mat", "lifetimeAlphaData", "-v7.3");

%% plot
save_dir = "D:\devs\inverted\processed_data\flim\mats\20210203_test_CR70A\images";

if exist("allXY", "var") == 0
    load("allXY.mat");
    load("allIntensityImages.mat");
    load("lifetimeImageData.mat");
    load("lifetimeAlphaData.mat");
end

AllnormalisationValue =[];
NormalisedAlphaData=[];
for i = 1:size(allIntensityImages, 1)
    ImagetoData =allIntensityImages{i};
    normalisationValue= max(max(ImagetoData));
    AllnormalisationValue(i) = normalisationValue;
    
    imageAlphaData = lifetimeAlphaData{i};
    for z = 1:Wavelength
        AlphaDataWavelength =imageAlphaData(z,:,:);
        normalisationValue= max(max(AlphaDataWavelength));
        NormalisedAlphaData(z,i) = normalisationValue;
    end
    
end
overallNormalisationValue = max(AllnormalisationValue);
overallNormalisationValue = overallNormalisationValue/scalingFactorIntensity; 

for i = 1:size(allXY, 1)
    mkdir(strcat(target_dir, "/", allXY{i}, "/images"));
    % intensity
    disp("    plot intensity image");
    ImagetoPlot =allIntensityImages{i};
    IntensityImagesNormalised=ImagetoPlot/overallNormalisationValue;
    folder = strcat(target_dir, "/", allXY{i}, "/images/intensity_normalised.jpg");
    lifetimeplotter(IntensityImagesNormalised, folder, NaN, 0, 1)

    disp("    plot lifetime image");
    a = 0;
    for z = firstSensorPixel:pixeldivider:lastSensorPixel
        a = a + 1;
        Currentwavelength = round(z*0.5468 + 500);
        
        OverallNormalisedAlphaFactor = max(NormalisedAlphaData(a,:));
        
        ImageData = lifetimeImageData{i};
        AlphaData = lifetimeAlphaData{i};
        
        ImagetoPlot = squeeze(ImageData(z, : , :));
        
        % lifetime only
%         folder = strcat(target_dir, "/", allXY{i}, "/images/lifetime_alpha_", num2str(Currentwavelength),'nm.jpg');
        folder = strcat(save_dir, "/lifetime_", allXY{i}, "_", num2str(Currentwavelength),'nm.jpg');
        lifetimeplotter(ImagetoPlot, folder, 1, 0.5, lifetimeHigh)
        
        % lifetime with alpha
%         AlphaData = squeeze(AlphaData(z,:,:));
%         AlphatoPlot = AlphaScalefactor*squeeze(AlphaData)/OverallNormalisedAlphaFactor;
% %         folder = strcat(target_dir, "/", allXY{i}, "/images/lifetime_alpha_", num2str(Currentwavelength),'nm.jpg');
%         folder = strcat(save_dir, "/lifetime_alpha_", allXY{i}, "_", num2str(Currentwavelength),'nm.jpg');
%         lifetimeplotter(ImagetoPlot, folder, AlphatoPlot, lifetimeLow, lifetimeHigh)
        
        % alpha-weighted lifetime
%         AlphatoPlot = AlphaScalefactor*squeeze(AlphaData)/OverallNormalisedAlphaFactor;
%         %folder = strcat(save_dir, "/", allXY{i}, "/images/lifetime_weighted_", num2str(Currentwavelength),'nm.jpg');
%         folder = strcat(save_dir, "/lifetime_weighted_", allXY{i}, "_", num2str(Currentwavelength),'nm.jpg');
% %         ImagetoPlotScaled = ImagetoPlot.*AlphatoPlot;
%         ImagetoPlotScaled = lifetime_blending(ImagetoPlot, AlphatoPlot);
%         lifetimeplotter(ImagetoPlotScaled, folder, 1, 0.05, lifetimeHigh)
        
        % lifetime with 1-alpha
%         AlphatoPlot = AlphaScalefactor*(1-(AlphaData/OverallNormalisedAlphaFactor));
%         folder = strcat(target_dir, "/", allXY{i}, "/images/lifetime_inverse_alpha_", num2str(Currentwavelength),'nm.jpg');
%         lifetimeplotter(ImagetoPlot, folder, AlphatoPlot, lifetimeLow, lifetimeHigh);
    end
end


%% plot per single exp
save_dir = "G:\inverted\processed\test";
to_dir = "G:\inverted\processed\20210209_CR70A_1\2_4x4_Row_2_col_2";
cd(to_dir)

% save("intensity.mat", "intensity_image");
% save("lifetime_cube.mat", "tauLeastSquaresReshaped");
% save("lifetime_alpha_cube.mat", "AlphaDataAll");
load("intensity.mat");
load("lifetime_alpha_cube.mat");
load("lifetime_cube.mat");

overallNormalisationValue = max(intensity_image);
overallNormalisationValue = overallNormalisationValue/scalingFactorIntensity; 

if ~exist(save_dir, 'dir')
       mkdir(save_dir);
end

disp("    plot lifetime image");
a = 0;
for z = firstSensorPixel:pixeldivider:lastSensorPixel
    a = a + 1;
    Currentwavelength = round(z*0.5468 + 500);
    
    OverallNormalisedAlphaFactor = max(max(max(AlphaDataAll)));
    
    ImageData = tauLeastSquaresReshaped;
    AlphaData = AlphaDataAll;
    
    ImagetoPlot = squeeze(ImageData(z, : , :));
    
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
    %         AlphatoPlot = AlphaScalefactor*(1-(AlphaData/OverallNormalisedAlphaFactor));
    %         folder = strcat(target_dir, "/", allXY{i}, "/images/lifetime_inverse_alpha_", num2str(Currentwavelength),'nm.jpg');
    %         lifetimeplotter(ImagetoPlot, folder, AlphatoPlot, lifetimeLow, lifetimeHigh);
end



%% blending
output_dir = "D:\devs\inverted\20190626_unstained_slidetissue\blended_images";
if exist("allXY", "var") == 0
    load("allXY.mat");
    load("allIntensityImages.mat");
    load("lifetimeImageData.mat");
    load("lifetimeAlphaData.mat");
end

blended_data = {};

for i = 1:size(allXY, 1)
    ImageData = lifetimeImageData{i};
    bgcolor = zeros(size(ImageData));
    AlphaData = lifetimeAlphaData{i};
    blended_data_wl = zeros(size(bgcolor));
    for f = 1:size(bgcolor, 1)
        image = squeeze(ImageData(f, :, :));
        alpha = squeeze(AlphaData(f, :, :));
        bg = squeeze(bgcolor(f, :, :));
        OverallNormalisedAlphaFactor = max(max(alpha));
        alpha = 1 - alpha/OverallNormalisedAlphaFactor;
        blended_data_wl(f, :, :) = (1-alpha).*bg + alpha.*image;
    end
    blended_data{i, 1} = blended_data_wl;
    
    save([allXY{i}, '_ia.mat'], 'blended_data_wl');
end

% blended_data_plot = blended_data{3, 1};
% load("F:\Backup\UoE\D_driver\data\Inverted\20190626_unstained_slidetissue\blended_images\x142_0_y15_5.mat");
% cm = [0 0 0;0 0 0.53125;0 0 0.546875;0 0 0.5625;0 0 0.578125;0 0 0.59375;0 0 0.609375;0 0 0.625;0 0 0.640625;0 0 0.65625;0 0 0.671875;0 0 0.6875;0 0 0.703125;0 0 0.71875;0 0 0.734375;0 0 0.75;0 0 0.765625;0 0 0.78125;0 0 0.796875;0 0 0.8125;0 0 0.828125;0 0 0.84375;0 0 0.859375;0 0 0.875;0 0 0.890625;0 0 0.90625;0 0 0.921875;0 0 0.9375;0 0 0.953125;0 0 0.96875;0 0 0.984375;0 0 1;0 0.015625 1;0 0.03125 1;0 0.046875 1;0 0.0625 1;0 0.078125 1;0 0.09375 1;0 0.109375 1;0 0.125 1;0 0.140625 1;0 0.15625 1;0 0.171875 1;0 0.1875 1;0 0.203125 1;0 0.21875 1;0 0.234375 1;0 0.25 1;0 0.265625 1;0 0.28125 1;0 0.296875 1;0 0.3125 1;0 0.328125 1;0 0.34375 1;0 0.359375 1;0 0.375 1;0 0.390625 1;0 0.40625 1;0 0.421875 1;0 0.4375 1;0 0.453125 1;0 0.46875 1;0 0.484375 1;0 0.5 1;0 0.515625 1;0 0.53125 1;0 0.546875 1;0 0.5625 1;0 0.578125 1;0 0.59375 1;0 0.609375 1;0 0.625 1;0 0.640625 1;0 0.65625 1;0 0.671875 1;0 0.6875 1;0 0.703125 1;0 0.71875 1;0 0.734375 1;0 0.75 1;0 0.765625 1;0 0.78125 1;0 0.796875 1;0 0.8125 1;0 0.828125 1;0 0.84375 1;0 0.859375 1;0 0.875 1;0 0.890625 1;0 0.90625 1;0 0.921875 1;0 0.9375 1;0 0.953125 1;0 0.96875 1;0 0.984375 1;0 1 1;0.015625 1 0.984375;0.03125 1 0.96875;0.046875 1 0.953125;0.0625 1 0.9375;0.078125 1 0.921875;0.09375 1 0.90625;0.109375 1 0.890625;0.125 1 0.875;0.140625 1 0.859375;0.15625 1 0.84375;0.171875 1 0.828125;0.1875 1 0.8125;0.203125 1 0.796875;0.21875 1 0.78125;0.234375 1 0.765625;0.25 1 0.75;0.265625 1 0.734375;0.28125 1 0.71875;0.296875 1 0.703125;0.3125 1 0.6875;0.328125 1 0.671875;0.34375 1 0.65625;0.359375 1 0.640625;0.375 1 0.625;0.390625 1 0.609375;0.40625 1 0.59375;0.421875 1 0.578125;0.4375 1 0.5625;0.453125 1 0.546875;0.46875 1 0.53125;0.484375 1 0.515625;0.5 1 0.5;0.515625 1 0.484375;0.53125 1 0.46875;0.546875 1 0.453125;0.5625 1 0.4375;0.578125 1 0.421875;0.59375 1 0.40625;0.609375 1 0.390625;0.625 1 0.375;0.640625 1 0.359375;0.65625 1 0.34375;0.671875 1 0.328125;0.6875 1 0.3125;0.703125 1 0.296875;0.71875 1 0.28125;0.734375 1 0.265625;0.75 1 0.25;0.765625 1 0.234375;0.78125 1 0.21875;0.796875 1 0.203125;0.8125 1 0.1875;0.828125 1 0.171875;0.84375 1 0.15625;0.859375 1 0.140625;0.875 1 0.125;0.890625 1 0.109375;0.90625 1 0.09375;0.921875 1 0.078125;0.9375 1 0.0625;0.953125 1 0.046875;0.96875 1 0.03125;0.984375 1 0.015625;1 1 0;1 0.984375 0;1 0.96875 0;1 0.953125 0;1 0.9375 0;1 0.921875 0;1 0.90625 0;1 0.890625 0;1 0.875 0;1 0.859375 0;1 0.84375 0;1 0.828125 0;1 0.8125 0;1 0.796875 0;1 0.78125 0;1 0.765625 0;1 0.75 0;1 0.734375 0;1 0.71875 0;1 0.703125 0;1 0.6875 0;1 0.671875 0;1 0.65625 0;1 0.640625 0;1 0.625 0;1 0.609375 0;1 0.59375 0;1 0.578125 0;1 0.5625 0;1 0.546875 0;1 0.53125 0;1 0.515625 0;1 0.5 0;1 0.484375 0;1 0.46875 0;1 0.453125 0;1 0.4375 0;1 0.421875 0;1 0.40625 0;1 0.390625 0;1 0.375 0;1 0.359375 0;1 0.34375 0;1 0.328125 0;1 0.3125 0;1 0.296875 0;1 0.28125 0;1 0.265625 0;1 0.25 0;1 0.234375 0;1 0.21875 0;1 0.203125 0;1 0.1875 0;1 0.171875 0;1 0.15625 0;1 0.140625 0;1 0.125 0;1 0.109375 0;1 0.09375 0;1 0.078125 0;1 0.0625 0;1 0.046875 0;1 0.03125 0;1 0.015625 0;1 0 0;0.984375 0 0;0.96875 0 0;0.953125 0 0;0.9375 0 0;0.921875 0 0;0.90625 0 0;0.890625 0 0;0.875 0 0;0.859375 0 0;0.84375 0 0;0.828125 0 0;0.8125 0 0;0.796875 0 0;0.78125 0 0;0.765625 0 0;0.75 0 0;0.734375 0 0;0.71875 0 0;0.703125 0 0;0.6875 0 0;0.671875 0 0;0.65625 0 0;0.640625 0 0;0.625 0 0;0.609375 0 0;0.59375 0 0;0.578125 0 0;0.5625 0 0;0.546875 0 0;0.53125 0 0;0.515625 0 0;0.5 0 0];
% for x = 1: size(blended_data_wl, 1)
%     imagesc(squeeze(blended_data_wl(x, :, :)));
%     colormap(cm);
%     caxis([0.05 lifetimeHigh]);
%     title(num2str(x));
%     colorbar;
%     pause(1/60);
% end
% imagesc(squeeze(mean(blended_data_wl)));
% save("blended_data.mat", "blended_data", "-v7.3");

%% test decay over timebins
%clear all;
load('G:\inverted\20210409_CR71A_4_FOV160_testing\HistMode_no_pixel_binning\2_3x3_Row_2_col_2\workspace.frame_1.mat');
starting = 120;
ending = 150;
dim = 256;
for x=starting:ending
    for y = starting:ending
        data = bins_array_3(:, :, x, y);
        data = mean(data);
        plot(data);
        title(['X: ', num2str(x), ', Y: ', num2str(y)])
        grid("on");
        pause(0.0001);
    end
end


%%
to_dir = "D:\devs\inverted\processed_data\flim\images\avg_intensity\";
global_max = 0.0;
for i=1:size(allIntensityImages, 1)
    image = allIntensityImages{i, 1};
    if global_max < max(max(image))
        global_max = max(max(image));
    end
end

cm = colormap(hot);
for i=1:size(allIntensityImages, 1)
    image = allIntensityImages{i, 1};
%     image = image/global_max;
%     image = histeq(image);
%     threshold = graythresh(image);
%     image(image < threshold) = 0;
    imagesc(image);
    axis("off");
    axis image;
    H = getframe(gca);
    imwrite(H.cdata, strcat(to_dir, num2str(i), ".png"));
end
