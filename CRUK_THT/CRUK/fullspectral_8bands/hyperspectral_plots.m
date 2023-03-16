%% added subtraction of data with too few counts by threashold - need to remove zeros from mean calculation

%% Calibration
excitation_wavelength = 485;
%slope_wavelength_vs_pixel = 0.729167;
%C = 520;
slope_wavelength_vs_pixel = 0.5468;
C = 500;
pixels = 1:1:512;
%wavelengths = zeros(512);
wavelengths = C + slope_wavelength_vs_pixel*pixels;
wavenumbers = zeros(512);
for i = 1:1:512
    wavelength_val = C + slope_wavelength_vs_pixel*i;
    wavenumbers(i) = (wavelength_val - 485)*10^7/(wavelength_val*485);
end

%% remove lifetime data with too few counts by ratio of bin 7 to bin 1 (peak to last selected)
selected_data_for_subtractionPeakbin = selected_data_for_subtraction(4,:);
selected_data_for_subtractionLongbin = selected_data_for_subtraction(1,:);
binRatioforSubtraction = selected_data_for_subtractionPeakbin./selected_data_for_subtractionLongbin;
binRatioforSubtraction(binRatioforSubtraction<6)=0;
binRatioforSubtraction(binRatioforSubtraction>0)=1;
parameters_cpu(2,:) = parameters_cpu(2,:).*binRatioforSubtraction;

%% remove lifetime data with too few counts by threashold on peak bin
selected_data_for_subtractionPeakbin = selected_data_for_subtraction(7,:);
binRatioforSubtraction = selected_data_for_subtractionPeakbin;
binRatioforSubtraction(binRatioforSubtraction<100)=0;
binRatioforSubtraction(binRatioforSubtraction>0)=1;
parameters_cpu(2,:) = parameters_cpu(2,:).*binRatioforSubtraction;
%% Plot pixel decay
spectral_pixel = 50;
row = 200;
col = 100;
bin_width = 0.8;
hyperspectralToolBox.plot_single_pixel_decay(bins_array_3, row, col, bin_width, spectral_pixel, wavelengths)
%% Plot intensity - bin counts
spectral_pixel = 100;
% Assumes we have already selected bins and subtracted mean Bin 16
bin = 2;
sample = 'Ahsan 13062019';
mask_threshold = 1;
array_movsum_selected = reshape(bins_array_movsum_selected_reshaped, size(bins_array_movsum_selected_reshaped, 2), size(bins_array_movsum_selected_reshaped, 1), size(bins_array_movsum_selected_reshaped, 3));
mask = hyperspectralToolBox.plot_bin_counts(mask_threshold, bin, spectral_pixel, array_movsum_selected, wavelengths, sample);
%% Plot lifetime image
spectral_pixel = 1;
convSize = 3;
sample = 'Normal';
bin = 2;
bins_array_alpha = reshape(array_movsum_selected(spectral_pixel, bin, :),[256 256]);
tauLeastSquaresCPU = parameters_cpu(2,:)';
tauLeastSquaresCPU(tauLeastSquaresCPU==0)=NaN;

hyperspectralToolBox.plot_lifetime_image_histogram_and_lifetime_v_wavelength(mask,bins_array_alpha, spectral_pixel,tauLeastSquaresCPU, wavelengths, sample, convSize)
%% RUN TILL HERE!
%%Differnece
%% Plot lifetime difference between lifetime images at different wavelengths
tauLeastSquaresReshaped = reshape(tauLeastSquaresCPU, [512 256 256]);
spectral_pixel_1 = 10
spectral_pixel_2 = 200
bin = 7
tauLeastSquaresReshapedDisplayFrame1 = reshape(tauLeastSquaresReshaped(spectral_pixel_1,:,:),[256 256]);
tauLeastSquaresReshapedDisplayFrame2 = reshape(tauLeastSquaresReshaped(spectral_pixel_2,:,:),[256 256]);
tauLeastSquaresReshapedDisplayFrameDifference = tauLeastSquaresReshapedDisplayFrame1 - tauLeastSquaresReshapedDisplayFrame2;
close all
bins_array_alpha = reshape(array_movsum_selected(spectral_pixel_1, bin, :),[256 256]);
%bins_array_alpha_normalised = 2.5*bins_array_alpha/max(max(bins_array_alpha));
bins_array_alpha_normalised = 1-bins_array_alpha/max(max(bins_array_alpha));
imagesc(tauLeastSquaresReshapedDisplayFrameDifference')
title('Lifetime Difference');
set(gcf,'color','w');
set(gca,'Color','k')
colorbar
caxis([0 1.3])
figure
imagesc(tauLeastSquaresReshapedDisplayFrameDifference', 'AlphaData', bins_array_alpha_normalised');
title('Lifetime Difference (Spectral Pixel 250 - 10) with Inverted Alpha');
set(gcf,'color','w');
set(gca,'Color','k')
colorbar
caxis([0 1.3])

%% Plot bin 16 sorted
spectral_pixel = 9;
hyperspectralToolBox.plot_bin_16_sorted(spectral_pixel, bin_16_array, wavelengths);
%% Plot lifetime histogram
spectral_pixel = 256;
hyperspectralToolBox.plot_histogram(spectral_pixel,tauLeastSquaresCPU, wavelengths);
%% Plot row and column versus wavelength
hyperspectralToolBox.plot_row_and_column_vs_wavelength(tauLeastSquaresCPU, 128, 128, wavelengths);
%% Plot Bin 16 mean
selectedSortPosition = 20;
global improvedBin16Mean;
hyperspectralToolBox.plot_bin16_mean(meanAllBins16VsWavelength, bin_16_array, selectedSortPosition, wavelengths);
%% Plot decays and Bin 16 mean
spectral_pixel = 50;
row = 236
col = 8
hyperspectralToolBox.plot_decays_and_bin16_mean(spectral_pixel, row, col, bins_array_3, wavelengths);
%% Plot decays and Bin 16 max

hyperspectralToolBox.plot_decays_and_bin16_mean(200, 128, 128, bins_array_3, wavelengths);

%% Plot lifetime images with row and column charts
spectral_pixel = 1;
hyperspectralToolBox.plot_lifetime_images_with_row_and_column_charts(spectral_pixel, tauLeastSquaresCPU, wavelengths);
%% Plot pixels
spectral_pixel = 200;
hyperspectralToolBox.plot_pixels(spectral_pixel, tauLeastSquaresCPU, wavelengths);
%% Plot and save lifetime images and histograms
hyperspectralToolBox.plot_and_save_lifetime_images_and_histograms(tauLeastSquaresCPU, wavelengths);
%% Bring plot of confocal pixels vs wavelength in here

