clc;clear;close all;
%% Paths
% base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-2_20230215/Mat_output';
% base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-9_20230222/Mat_output';
% base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-13_20230226/Mat_output';
% % base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-3_Col-5_20230218/Mat_output2';
% base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/RT/Row-4_Col-1_20230214/Mat_output2';
% base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/RT/Row-6_Col-10_20230223/Mat_output2';

base_dir_all=cell(5,1);
base_dir_all{1,1}='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-2_20230215/Mat_output';
base_dir_all{2,1}='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-9_20230222/Mat_output';
base_dir_all{3,1}='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-13_20230226/Mat_output';
% base_dir_all='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-3_Col-5_20230218/Mat_output2';
base_dir_all{4,1}='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/RT/Row-4_Col-1_20230214/Mat_output2';
base_dir_all{5,1}='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/RT/Row-6_Col-10_20230223/Mat_output2';
%% Training data generation
matfilename='feat_flt_1to4.mat';
% matfilename='feat_dist_1to4.mat';
% matfilename='feat_aug_23.mat';
base_dir=base_dir_all{1,1};
cd(base_dir)
% load('feat_flt_1.mat');
load(matfilename);
% disp(size(class_data_1));

% feat_matrix1=feat_matrix2;
% class_grnd_trth1=class_grnd_trth2;

% train_data=class_data_1;
train_data=feat_matrix1;
train_label=class_grnd_trth1;

for base_i = 2:length(base_dir_all)
    base_dir=base_dir_all{base_i,1};
    cd(base_dir)
    % load('feat_flt_1.mat');
    load(matfilename);
    % disp(size(class_data_1));
    % train_data=[train_data;class_data_1];
    % feat_matrix1=feat_matrix2;
    % class_grnd_trth1=class_grnd_trth2;
    
    train_data=[train_data;feat_matrix1];
    train_label=[train_label;class_grnd_trth1];
end
