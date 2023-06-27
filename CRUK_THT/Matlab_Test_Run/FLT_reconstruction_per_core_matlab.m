
%%test analysis script
clc;clear;close all;
%% Parameters
nopeaks=3;
no_of_spectral_channels=310;
numberoflambdas=no_of_spectral_channels;
binToFit=[12,15];
binToFit2=[1,15];
% outputdir='C:\Users\CRUK EDD\Documents\MATLAB\Test_Output\';
%% FLT reconstruction
currentFolder = pwd;
% filePath = uigetdir; 
filePath = '/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-2_20230215/';

flt_recon_core_mat(filePath,currentFolder,numberoflambdas,binToFit);
%%

% completed_time=toc/60;