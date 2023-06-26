%% "Intensity Image" THIS ONE!!!!!
figure
frame_size_x =256;
wholeImageBnd1=zeros(frame_size_x);
wholeImageBnd2=zeros(frame_size_x);
for bin = 1:13
    
    disp(bin)
    spectral_bins_per_pixel_mesh = bins_array_3(:,bin,:);
    spectral_bins_per_pixel_mesh_reshaped = reshape(spectral_bins_per_pixel_mesh,[512 frame_size_x frame_size_x]);
    
    for spectral_pixel = 1:150
        spectral_bins_per_pixel_mesh_reshapedoneFrame = reshape(spectral_bins_per_pixel_mesh_reshaped(spectral_pixel,:,:), [frame_size_x frame_size_x]);
        wholeImageBnd1 = wholeImageBnd1+spectral_bins_per_pixel_mesh_reshapedoneFrame;
        
    end
    
end

for bin = 1:13
    
    disp(bin)
    spectral_bins_per_pixel_mesh = bins_array_3(:,bin,:);
    spectral_bins_per_pixel_mesh_reshaped = reshape(spectral_bins_per_pixel_mesh,[512 frame_size_x frame_size_x]);
    
    for spectral_pixel = 300:500
        spectral_bins_per_pixel_mesh_reshapedoneFrame = reshape(spectral_bins_per_pixel_mesh_reshaped(spectral_pixel,:,:), [frame_size_x frame_size_x]);
        wholeImageBnd2 = wholeImageBnd2+spectral_bins_per_pixel_mesh_reshapedoneFrame;
        
    end
    
end


% 
% wholeImageBnd1(wholeImageBnd1<100) =0;
% wholeImageBnd2(wholeImageBnd2>1000) =0;
figure
imagesc(wholeImageBnd1');
axis off
%ti = get(gca,'TightInset')
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.3, 0.3, 0.3, 0.50]);
ti = [ 0 0 0 0.09 ];
set(gca,'Position',[ti(1) ti(2) 1-ti(3)-ti(1) 1-ti(4)-ti(2)]);
set(gca,'units','centimeters')
pos = get(gca,'Position');
%ti = get(gca,'TightInset');
colorbar
colormap(hot)
% caxis([0 80000])
set(gcf, 'PaperUnits','centimeters');
set(gcf, 'PaperSize', [pos(3)+ti(1)+ti(3) pos(4)+ti(2)+ti(4)]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition',[0 0 pos(3)+ti(1)+ti(3) pos(4)+ti(2)+ti(4)]);

figure
imagesc(wholeImageBnd2');
axis off
%ti = get(gca,'TightInset')
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.3, 0.3, 0.3, 0.50]);
ti = [ 0 0 0 0.09 ];
set(gca,'Position',[ti(1) ti(2) 1-ti(3)-ti(1) 1-ti(4)-ti(2)]);
set(gca,'units','centimeters')
pos = get(gca,'Position');
%ti = get(gca,'TightInset');
colorbar
colormap(hot)
% caxis([0 80000])
set(gcf, 'PaperUnits','centimeters');
set(gcf, 'PaperSize', [pos(3)+ti(1)+ti(3) pos(4)+ti(2)+ti(4)]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition',[0 0 pos(3)+ti(1)+ti(3) pos(4)+ti(2)+ti(4)]);





%colormap(hot);


%% Plot Spectrum for a pixel - USE THIS ONE TOO
spectral_bins_per_pixel = reshape(bins_array_3,[512,16,frame_size_x,frame_size_x]);
pix_x = 142;
pix_y = 83;
spectral_bins_per_pixel_pix_x_y = spectral_bins_per_pixel(:,:,pix_x,pix_y);
%spectral_bins_per_pixel_pix_x_y = bins_array(:,:,64);
bin = 14;
spectral_bins_per_pixel_pix_x_y_bin = spectral_bins_per_pixel_pix_x_y(:,bin);
figure;
plot(movmean(spectral_bins_per_pixel_pix_x_y_bin,1))
%xlim([0 200])
%ylim([0 1000])
% figure;
% smoothedData = movmean(spectral_bins_per_pixel_pix_x_y_bin,5);
% plot(smoothedData);


%% Subtracted Spectrum
spectral_bins_per_pixel = reshape(bins_array_3,[512,16,frame_size_x,frame_size_x]);
pix_x = 136;
pix_y = 235;
backgroundBin = 1;
ramanBin = 14;
spectral_bins_per_pixel_pix_x_y = spectral_bins_per_pixel(:,:,pix_x,pix_y);

background =  spectral_bins_per_pixel_pix_x_y(:,backgroundBin);
spectral_bins_per_pixel_pix_x_y_bin = spectral_bins_per_pixel_pix_x_y(:,ramanBin);

subtractedSpectra = spectral_bins_per_pixel_pix_x_y_bin-background;
background =  spectral_bins_per_pixel_pix_x_y(:,backgroundBin);

spectral_bins_per_pixel_pix_x_y_bin = spectral_bins_per_pixel_pix_x_y(:,ramanBin);
subtractedSpectra2 = spectral_bins_per_pixel_pix_x_y_bin-background;

spectral_bins_per_pixel_pix_x_y = spectral_bins_per_pixel(:,:,pix_x+4,pix_y+4);
background =  spectral_bins_per_pixel_pix_x_y(:,backgroundBin);

spectral_bins_per_pixel_pix_x_y_bin = spectral_bins_per_pixel_pix_x_y(:,ramanBin);
subtractedSpectra3 = spectral_bins_per_pixel_pix_x_y_bin-background;


figure;
hold on
plot(movmean(subtractedSpectra,1))
plot(movmean(subtractedSpectra2,1))
plot(movmean(subtractedSpectra3,1))
xlim([0 200])
hold off



%% Plot decay for 2 sensor pixels for the selected image pixel -  YAY THIS ONE!
figure;
sensorPixel = 150;
decay1 = spectral_bins_per_pixel_pix_x_y(sensorPixel,:);
decay2 = spectral_bins_per_pixel_pix_x_y(sensorPixel+10,:);
plot(decay1);
hold;
plot(decay2);
%% Mesh of spectra vs sensor pixel - DONT USE THIS
sum_spectra_for_all_bins = zeros(0);
numberofPixels = frame_size_x^2;

all_spectra = zeros(512, numberofPixels);
for bin = 1:16
%all_spectra = zeros(0);
disp(bin);
for pix = 1:numberofPixels-1
    spectral_bins_per_pixel = bins_array_3(:,:,pix);
    all_spectra(:, pix) = movmean(spectral_bins_per_pixel(:,bin),10);
end
sum_spectra = sum(all_spectra,2);
sum_spectra_for_all_bins = [sum_spectra_for_all_bins sum_spectra];
end

%% PLOT Pixel Mesh
sum_spectra_for_all_bins = zeros(0);
numberofPixels = 1024;

all_spectra = [];
for bin = 1:16
%all_spectra = zeros(0);
disp(bin);
for pix = 1:numberofPixels
    spectral_bins_per_pixel = bins_array_3(:,:,pix);
    all_spectra(:, pix) = movmean(spectral_bins_per_pixel(:,bin),10);
end
sum_spectra = sum(all_spectra,2);
sum_spectra_for_all_bins = [sum_spectra_for_all_bins sum_spectra];
end

%% DONT USE THIS
figure
mesh(sum_spectra_for_all_bins)
%set(gca, 'ZScale', 'log')
%zlim([500000 100000000])
ylim([0,512]);
xlim([4 16]);
title('Pixel 65x / 37y', 'FontSize', 26);
%set(gca,'Xtick',1000:1000:4000)
%set(gca,'Ytick',1000:1000:4000)
%set(gca,'Ztick',1000:1000:102000)
    


%% DONT USE THIS
close all
 plot(sum_spectra_for_all_bins(:,14))
 ylim([0 300000]);
%% subtract bin 16 DONT USE THIS 
figure;
BaseData = sum_spectra_for_all_bins;

allBins16 = bins_array_3(:,16,:);
allBins16Reshaped = reshape(allBins16, [512 65536]);
meanAllBins16VsWavelength = mean(allBins16Reshaped,2);
meanAllBins16VsWavelength = meanAllBins16VsWavelength.*65536;

for bin = 1:16
BaseData(:,bin)  = BaseData(:,bin) - BaseData(:,16);
end
mesh(BaseData);
%ylim([0 400]);
xlim([2 15]);
%zlim([0 4000000]);
% figure;
% %plot(sum_spectra_for_all_bins(100,:));
% plot(BaseData(:,16));
figure;
imagesc(BaseData);

figure;
hold on
for i = 1:50:300
plot(log(BaseData(i,2:16)));
end
hold off
%% DONT USE THIS ONE EIthER
numberOfBins = 10;
timeBins = (0:1:numberOfBins)*0.8;
timeBins = timeBins';
timeBins2 = [ones(length(timeBins),1) timeBins];

tauAllwavelengths = zeros(512,1);
for pixel =1:512

logCounts = log(BaseData(pixel,3:13));
fit = timeBins2\logCounts';
tauAllwavelengths(pixel) = real(1/fit(2));

end
figure;
plot(tauAllwavelengths);
ylim([0 6]);




%% DONT USE THIS ONE (GETTING SILLY NOW)
figure
[pks,locs] = findpeaks(sum_spectra_for_all_bins(:,13));
[pks2,locs2] = findpeaks(sum_spectra_for_all_bins(:,12));
difference_v_bin16 = sum_spectra_for_all_bins(:,12) - sum_spectra_for_all_bins(:,16);
%plot(sum_spectra_for_all_bins(:,11))
plot(difference_v_bin16)
hold
%plot(sum_spectra_for_all_bins(:,15))
%plot(sum_spectra_for_all_bins(:,14))
%ylim([0 500000])
title('4096 pixel summed spectra Bins 13 showing common peaks with bin 12')
xlabel('Pixel/wavelength')
ylabel('Counts')
set(gcf,'color','w');
%stem(locs, pks)
%stem(locs2, pks2)
common_peaks = intersect(locs, locs2);
ones_peaks = 12000000*ones(87,1);
%stem(common_peaks,ones_peaks)
%% NOT THIS ONE EITHER
mesh(all_spectra)
figure
plot(sum_spectra)
%% Single frame wavelength / bin - OK YOU GET THE POINT! NOT THIS ONE
close all
bin = 10;
spectral_pixel = 100;
spectral_bins_per_pixel_mesh = bins_array_3(:,bin,:);
spectral_bins_per_pixel_mesh_reshaped = reshape(spectral_bins_per_pixel_mesh,[512 256 256]);
spectral_bins_per_pixel_mesh_reshapedoneFrame = spectral_bins_per_pixel_mesh_reshaped(spectral_pixel,:,:);
spectral_bins_per_pixel_mesh_reshapedoneFrame = reshape(spectral_bins_per_pixel_mesh_reshapedoneFrame,[256 256]);
imagesc(spectral_bins_per_pixel_mesh_reshapedoneFrame)
colorbar
figure
row = 66;
col = 184;
spectrum_one_confocal_pixel = spectral_bins_per_pixel_mesh_reshaped(:,row,col);
plot(movmean(spectrum_one_confocal_pixel,5))




%% Image for  given bin and spectral pixel - OH NO NOT THIS ONE
spectral_pixel = 10;
bin = 13;
spectral_image_for_bin = reshape(spectral_bins_per_pixel(spectral_pixel,bin,:,:),[64,64]);
imagesc(spectral_image_for_bin)


%% rowplot
row = 12;
colum = 16
bin = 14;
rowData = bins_array_3(:,bin,row,:);
columData = bins_array_3(:,bin,:,colum);


spectral_bins_row = reshape(rowData,[512,32]);
spectral_bins_colum = reshape(columData,[512,32]);
figure
imagesc(spectral_bins_row);
ylim([0 500])
figure
imagesc(spectral_bins_colum);
ylim([0 500])


