clc;clear; close all;
%%
% load('C:\Users\THT_CRUK\Documents\MATLAB\Test_Data\spectral_analysis_load_512_A.mat');
load('C:\Users\THT_CRUK\Documents\MATLAB\Test_Data\512Band_data\HistMode_no_pixel_binning\5_3x3_Row_1_col_1\workspace.frame_1.mat','bins_array_3');
%%
% bin_array=Int_Prof{1,3};    
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
acc_arr1a=mean(acc_arr1);
acc_arr1b=mean(acc_arr1,2);

acc_arr1La=mean(acc_arr1L);
acc_arr1Lb=mean(acc_arr1L,2);

acc_arr2La=mean(acc_arr2L);
acc_arr2Lb=mean(acc_arr2L,2);

acc_arr3La=mean(acc_arr3L);
acc_arr3Lb=mean(acc_arr3L,2);

acc_arr4La=mean(acc_arr4L);
acc_arr4Lb=mean(acc_arr4L,2);

acc_arr5La=mean(acc_arr5L);
acc_arr5Lb=mean(acc_arr5L,2);
%% Visualisation

figure(1);
subplot(2,2,1),
plot(acc_arr1a);
xlabel('time');
ylabel('intensity')
title('Time vs Intensity Across Spatial Dimension');

% figure(2),
subplot(2,2,2),
plot(acc_arr1La);hold all;
plot(acc_arr2La);
plot(acc_arr3La);
plot(acc_arr4La);
plot(acc_arr5La);hold off;
legend({'1st quarter','2nd quarter','3rd quarter','4th quarter','Centre'},'location','Northwest');
xlabel('time');
ylabel('intensity')
title('Time vs Intensity at specific coordinates');

% figure(3);
subplot(2,2,3),
plot(acc_arr1b);
xlabel('time');
ylabel('intensity')
title('Intensity spectrum Across Spatial Dimension');

% figure(4),
subplot(2,2,4),
plot(acc_arr1Lb);hold all;
plot(acc_arr2Lb);
plot(acc_arr3Lb);
plot(acc_arr4Lb);
plot(acc_arr5Lb);hold off;
legend({'1st quarter','2nd quarter','3rd quarter','4th quarter','Centre'},'location','Northwest');
xlabel('time');
ylabel('intensity')
title('Intensity spectrum at specific coordinates');