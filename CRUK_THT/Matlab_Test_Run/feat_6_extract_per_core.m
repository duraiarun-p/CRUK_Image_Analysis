clc;clear;close all;
%% Paths
base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-2_20230215/Mat_output';
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

% base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/Row-1_Col-9_20230222/FLT_IMG_DIR/Stitched';

for base_i = 1:length(base_dir_all)
% for base_i = 1:1
%     prepare_data(base_dir_all{base_i,1});
% end

% function prepare_data(base_dir)
base_dir=base_dir_all{base_i,1};
%% Classification file extraction 
cd(base_dir)

%% Loading files
HE=imread('coreg_HE.tiff');
% Load Classification ground truth
load('data_gt.mat');
% Load FLIM images
load("core_stitched_masked_TX.mat")

%%
flt_int=stitch_flt_cube_masked;
flt_siz=size(flt_int);
HE_siz=size(HE);

nooffeat=500;
class_1_ind=find(class_grnd_trth==1);
class_2_ind=find(class_grnd_trth==2);
class_3_ind=find(class_grnd_trth==3);

% class_1=class_grnd_trth(class_1_ind(1:nooffeat*2));
% class_2=class_grnd_trth(class_2_ind(1:nooffeat));
% class_3=class_grnd_trth(class_3_ind(1:nooffeat));

%%
noofcells=length(bound_txed);
% feat_matrix=zeros(nbin*flt_siz(3),noofcells);
% % feat_matrix=zeros(flt_siz(3)*nbin,noofcells);
cells_img=cell(noofcells,1);
for cell_ind=1:noofcells
% for cell_ind=1:1
% row_start=floor(bound_txed(cell_ind,1));
% row_stop=floor(bound_txed(cell_ind,3));
% col_start=floor(bound_txed(cell_ind,2));
% col_stop=floor(bound_txed(cell_ind,4));

row_start=max(floor(bound_txed(cell_ind,1)),1);
row_stop=min(floor(bound_txed(cell_ind,3)),HE_siz(1));
col_start=max(floor(bound_txed(cell_ind,2)),1);
col_stop=min(floor(bound_txed(cell_ind,4)),HE_siz(2));

cell_box=flt_int(row_start:row_stop,col_start:col_stop,:);
cells_img{cell_ind,1}=cell_box;

end

nooffeat=500;
tumour_times=1;

class_T=class_grnd_trth(class_1_ind(1:nooffeat*tumour_times));
class_S=class_grnd_trth(class_2_ind(1:nooffeat));
class_I=class_grnd_trth(class_3_ind(1:nooffeat));

cells_T=cells_img(class_1_ind(1:nooffeat*tumour_times));
cells_S=cells_img(class_2_ind(1:nooffeat));
cells_I=cells_img(class_3_ind(1:nooffeat));

save('cells_img.mat','cells_I','cells_S','cells_T','class_I','class_S','class_T');

end