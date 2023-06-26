%% Spectral Analysis of LFS and SEXP Fitting
clc;clear;close all;
%%
currentFolder = pwd;
%%
%  load('C:\Users\THT_CRUK\Documents\MATLAB\Test_Data\512Band_data\HistMode_no_pixel_binning\5_3x3_Row_2_col_2\workspace.frame_1.mat');
 load('C:\Users\THT_CRUK\Documents\MATLAB\Test_Data\Row-5_Col-1_20230210\Row_2_Col_2\workspace.frame_1.mat')
 %%
 no_of_spectral_channels=310;

 myplotspectrum(bins_array_3,5,6);

bins_array_3=bins_array_3(1:no_of_spectral_channels,:,:,:);

myplotspectrum(bins_array_3,7,8);

 %%
 [allIntensityImages,lifetimeImageData,lifetimeAlphaData]=Analysis_LMfitting_per_tile(bins_array_3,[12,15],HIST_MODE,frame_size_x);
%%
n_bins=16;
nspectrum_split=16;
 % Extraction of bin_array_3 dimension's info
siz=size(bins_array_3);

% spatial_dimension_ind=find(siz==frame_size_x); % to permutate the array in the right order and maintain consistency
% temporal_dimension_ind=find(siz==n_bins);
% remain_ind=union(temporal_dimension_ind,spatial_dimension_ind);
% spectral_dimension_ind=find(siz);
% spectral_dimension_ind(remain_ind)=[];
% n_spectrum = siz(spectral_dimension_ind);

temporal_dimension_ind=find(siz==n_bins);
spatial_dimension_ind=[temporal_dimension_ind+1,temporal_dimension_ind+2];
spectral_dimension_ind=temporal_dimension_ind-1;

n_spectrum = siz(spectral_dimension_ind);


 allIntensityImages=squeeze(allIntensityImages);

 plot_spectrum=1:nspectrum_split:n_spectrum;
 plotspectrumL=length(plot_spectrum);
 lifetimeImageData_selected=lifetimeImageData(plot_spectrum,:,:);

 lifetimeAlphaData_selected=lifetimeAlphaData(plot_spectrum,:,:);

%  clim_max1=max(lifetimeImageData(:));
%  clim_min2=min(lifetimeImageData(:));



 allIntensityImages_filtered=medfilt2(allIntensityImages);
%%
figure(1);subplot(121);
imagesc(allIntensityImages);
title('allintensityImage - jet');
subplot(122);
imshow(allIntensityImages,[]);
title('allintensityImage - gray');

%%
figure(10);
imshow(allIntensityImages,[]);colorbar;
title('allintensityImage');
%%

figure(2);subplot(121);
imagesc(allIntensityImages_filtered);
title('allIntensityImages_{filtered} - jet');
subplot(122);
imshow(allIntensityImages_filtered,[]);
title('allIntensityImages_{filtered} - gray');

%%
plotrow=round(sqrt(plotspectrumL));
plotcol=round(plotspectrumL/plotrow);
clim_max=4;
 clim_min=1.0;
figure(3);
for plot_index=1:plotspectrumL
subplot(plotrow,plotcol,plot_index)
imagesc(squeeze(lifetimeImageData_selected(plot_index,:,:)));
clim([clim_min clim_max])
set(gca,'Color','k');
colormap jet;
subtitle(['\lambda_{',num2str(plot_spectrum(plot_index)),'}']);
end


%%
% clim_max1=4000;
%  clim_min1=0;
 clim_max1=max(lifetimeAlphaData_selected(:));
 clim_min1=min(lifetimeAlphaData_selected(:));
figure(4);
for plot_index=1:plotspectrumL
subplot(plotrow,plotcol,plot_index)
imagesc(squeeze(lifetimeAlphaData_selected(plot_index,:,:)));
clim([clim_min1 clim_max1])
set(gca,'Color','k');
colormap jet;
subtitle(['\lambda_{',num2str(plot_spectrum(plot_index)),'}']);
end
%title('lifetimeImageData');

%%
clim_min1=1;
clim_max2=4;
figure(30);
for plot_index=1:plotspectrumL
subplot(plotrow,plotcol,plot_index)
h=imagesc(squeeze(lifetimeImageData_selected(plot_index,:,:)));
clim([clim_min1 clim_max2])
I=squeeze(lifetimeAlphaData_selected(plot_index,:,:));
In=I/max(I(:));
In=In*2.5;
set(h, 'AlphaData', In);
set(gca,'Color','k');
colormap jet;
subtitle(['\lambda_{',num2str(plot_spectrum(plot_index)),'}']);
end
%%
function myplotspectrum(bins_array_3,fignum,fignum1)
[spectrums,times,xs,ys]=size(bins_array_3);

acc_arr1=zeros(spectrums,times);
acc_arr1L=zeros(spectrums,times);
acc_arr2L=zeros(spectrums,times);
acc_arr3L=zeros(spectrums,times);
acc_arr4L=zeros(spectrums,times);
acc_arr5L=zeros(spectrums,times);

for spec_1=1:spectrums
    for time_1=1:times
        bin=squeeze(bins_array_3(spec_1,time_1,:,:));
        acc_arr1(spec_1,time_1)=sum(bin(:));
        acc_arr1L(spec_1,time_1)=bin(xs*0.25,ys*0.25);
        acc_arr2L(spec_1,time_1)=bin(xs*0.75,ys*0.25);
        acc_arr3L(spec_1,time_1)=bin(xs*0.75,ys*0.25);
        acc_arr4L(spec_1,time_1)=bin(xs*0.75,ys*0.75);
        acc_arr5L(spec_1,time_1)=bin(xs*0.5,ys*0.5);

    end
end
%%
acc_arr1a=(mean(acc_arr1));
acc_arr1b=smooth(mean(acc_arr1,2));

acc_arr1La=(mean(acc_arr1L));
acc_arr1Lb=smooth(mean(acc_arr1L,2));

acc_arr2La=(mean(acc_arr2L));
acc_arr2Lb=smooth(mean(acc_arr2L,2));

acc_arr3La=(mean(acc_arr3L));
acc_arr3Lb=smooth(mean(acc_arr3L,2));

acc_arr4La=(mean(acc_arr4L));
acc_arr4Lb=smooth(mean(acc_arr4L,2));

acc_arr5La=(mean(acc_arr5L));
acc_arr5Lb=smooth(mean(acc_arr5L,2));
%%
figure(fignum),
plot(acc_arr1Lb);hold all;
plot(acc_arr2Lb);
plot(acc_arr3Lb);
plot(acc_arr4Lb);
plot(acc_arr5Lb);hold off;
legend({'1st quarter','2nd quarter','3rd quarter','4th quarter','Centre'});
xlabel('spectrum');
ylabel('intensity')
title('Intensity spectrum at specific coordinates');

figure(fignum1),
plot(acc_arr1La);hold all;
plot(acc_arr2La);
plot(acc_arr3La);
plot(acc_arr4La);
plot(acc_arr5La);hold off;
legend({'1st quarter','2nd quarter','3rd quarter','4th quarter','Centre'},'location','Northwest');
xlabel('time');
ylabel('intensity')
title('Time vs Intensity at specific coordinates');
end
%%
% acc_arr6L=zeros(spectrums,times);
% for spec_1=1:spectrums
%     for time_1=1:times
%         bin=squeeze(bins_array_3(spec_1,time_1,:,:));
% %         acc_arr1(spec_1,time_1)=sum(bin(:));
%         acc_arr6L(spec_1,time_1)=bin(217,471);
%     end
% end
% acc_arr6La=(mean(acc_arr6L));
% acc_arr6Lb=smooth(mean(acc_arr6L,2));