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
%% 
currentworkingFolder = fileparts(mfilename('fullpath'));
core_Path = '/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/';
core_Folder=dir(fullfile(core_Path, '*Row*'));
core_Folder(1:4)=[];
core_Folder_len=length(core_Folder);
%%
for core_i = 1 : core_Folder_len
    
    core_Folder_File=strcat(core_Folder(core_Folder_len).folder,'/',core_Folder(core_i).name,'/');

    disp(core_Folder_File)
    flt_recon_core_mat(core_Folder_File,currentworkingFolder,numberoflambdas,binToFit);

    cd(currentworkingFolder)
end