
%%test analysis script
clc;clear;close all;
addpath('/home/cruk/Documents/Gpufit-build/matlab/');
addpath('/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/CRUK_THT/Matlab_Test_Run/fullspectral_8bands/');
%% Parameters
nopeaks=3;
no_of_spectral_channels=310;
numberoflambdas=no_of_spectral_channels;
binToFit=[12,15];
binToFit2=[1,15];
% outputdir='C:\Users\CRUK EDD\Documents\MATLAB\Test_Output\';
%% FLT reconstruction
% currentFolder = pwd;
currentFolder='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/FLT_Output/';
% filePath = uigetdir; 
% filePath = '/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-2_Col-3_20230216';
% filePath = '/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-2_20230215';
% filePath='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-3_Col-5_20230218';
% filePath='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/RT/Row-4_Col-1_20230214';
% % filePath='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/RT/Row-6_Col-10_20230223';

% core_lists=dir('/run/user/1000/gvfs/smb-share:server=cmvm.datastore.ed.ac.uk,share=cmvm/scs/groups/kdhaliwa-CRUKEDD/TMA/FS-FLIM/raw/Tumour_2B/');

core_lists=dir('//mnt/local_share/TMA/FS-FLIM/raw/Tumour_2B/');
core_lists(1:2)=[];

core_number=14;
core_number=core_number+1;
% for core_number=5:length(core_lists)
filePath=[core_lists(core_number).folder,'/',core_lists(core_number).name];
disp(filePath);
% oP_Folder=[currentFolder,core_lists(core_number).name,'/'];
% 
% if not(isfolder(oP_Folder))
%     mkdir(oP_Folder)
% end


%%
tic;
% flt_recon_core_mat_1(filePath,oP_Folder,numberoflambdas,binToFit2);
flt_recon_core_mat(filePath,currentFolder,numberoflambdas,binToFit2);
%%
disp(filePath);
completed_time=toc/60;
% end