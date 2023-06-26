%% Spectral Analysis of Tumour vs Normal
clc;clear;close all;
%%
currentFolder = pwd;
%%
%  load('C:\Users\THT_CRUK\Documents\MATLAB\Test_Data\512Band_data\HistMode_no_pixel_binning\5_3x3_Row_2_col_2\workspace.frame_1.mat');
 load('C:\Users\THT_CRUK\Documents\MATLAB\Test_Data\Row-1_Col-1_20230303_N\Row_2_Col_2\workspace.frame_1.mat')
 %%
%  no_of_spectral_channels=310;
[xs,ys,times,spectrums]=size(bins_array_3);

Spec_resp1=zeros(spectrums,1);

for spect_index=1:spectrums
    bin_resp=bins_array_3(:,:,:,spect_index);
    Spec_resp1(spect_index)=sum(bin_resp(:));
end
clear bins_array_3 bin_resp
load ('C:\Users\THT_CRUK\Documents\MATLAB\Test_Data\Row-1_Col-1_20230214_TM2B\Row_2_Col_2\workspace.frame_1.mat')
Spec_resp2=zeros(spectrums,1);

for spect_index=1:spectrums
    bin_resp=bins_array_3(:,:,:,spect_index);
    Spec_resp2(spect_index)=sum(bin_resp(:));
end

clear bins_array_3 bin_resp