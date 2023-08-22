clc;clear;close all;
%% Paths
base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-2_20230215/Mat_output';
% base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-9_20230222/Mat_output';
% base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-13_20230226/Mat_output';
% base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-3_Col-5_20230218/Mat_output2';
% base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/RT/Row-4_Col-1_20230214/Mat_output2';
% base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/RT/Row-6_Col-10_20230223/Mat_output2';
%% Classification file extraction 
cd(base_dir)

%% Loading files
HE=imread('coreg_HE.tiff');
% Load Classification ground truth
load('data_gt.mat');
% Load FLIM images
load("core_stitched_masked.mat")
%%

% u8_flt=uint8(zeros(size(stitch_flt_cube_masked)));
% u8_int=uint8(zeros(size(stitch_intensity_cube_masked)));
flt_int=uint8(zeros(size(stitch_intensity_cube_masked)));
flt_siz=size(stitch_intensity_cube_masked);
% spec=38;
for spec=1:flt_siz(3)
slice_flt=stitch_flt_cube_masked(:,:,spec);
slice_int=stitch_intensity_cube_masked(:,:,spec);
% slice_flt_u8=uint8(255*mat2gray(slice_flt));
% slice_int_u8=uint8(255*mat2gray(slice_int));
slice_flt_u8=(1*mat2gray(slice_flt));
slice_int_u8=(1*mat2gray(slice_int));

[new_slice, new_d]=imfuse(slice_flt_u8,slice_int_u8,'blend');
% flt_int(:,:,spec)=new_slice;% Intensity weighted lifetime
flt_int(:,:,spec)=slice_flt_u8;%Lifetime
end

%%

figure(7),
imshow(HE)
%%
nbin=256;
nopcs=50;
new_cell_width=20;
cell_ind=1;
no_of_classes=3;
noofcells=length(bound_txed);
% feat_matrix=zeros(nbin*flt_siz(3),noofcells);
feat_matrix=zeros(flt_siz(3)*new_cell_width*new_cell_width,noofcells);
cells_img=cell(noofcells,2);
for cell_ind=1:noofcells
% for cell_ind=1:1
% row_start=floor(bound_txed(cell_ind,1));
% row_stop=floor(bound_txed(cell_ind,3));
% col_start=floor(bound_txed(cell_ind,2));
% col_stop=floor(bound_txed(cell_ind,4));

row_start=max(floor(bound_txed(cell_ind,1)),1);
row_stop=max(floor(bound_txed(cell_ind,3)),1);
col_start=max(floor(bound_txed(cell_ind,2)),1);
col_stop=max(floor(bound_txed(cell_ind,4)),1);

cell_box=flt_int(row_start:row_stop,col_start:col_stop,:);
cell_box=reshape(imresize3(cell_box,[new_cell_width,new_cell_width,flt_siz(3)]),new_cell_width*new_cell_width,flt_siz(3));
cell_size=size(cell_box);
cells_img{cell_ind,1}=cell_box(:);
% cells_img{cell_ind,2}=class_grnd_trth(cell_ind);
cells_img{cell_ind,2}=int2bit(class_grnd_trth(cell_ind),no_of_classes);
% disp(cell_size);
feat_matrix(:,cell_ind)=cell_box(:);
end

%%
% dataSource = groundTruthDataSource(cells_img);
grnd_truth=cell2table(cells_img);
% grnd_truth(:,2)=class_grnd_trth;
grnd_truth.Properties.VariableNames={'cells_img'  'labels'};

gnd_truth_ds=arrayDatastore(grnd_truth,"OutputType","same");
%%
nooffeat=500;
class_1_ind=find(class_grnd_trth==1);
class_2_ind=find(class_grnd_trth==2);
class_3_ind=find(class_grnd_trth==3);

class_1=class_grnd_trth(class_1_ind(1:nooffeat*2));
class_2=class_grnd_trth(class_2_ind(1:nooffeat));
class_3=class_grnd_trth(class_3_ind(1:nooffeat));

% feat_1=feat_matrix(class_1_ind(1:1000),:);
% feat_2=feat_matrix(class_2_ind(1:1000),:);
% feat_3=feat_matrix(class_3_ind(1:1000),:);

feat_1=feat_matrix(:,class_1_ind(1:nooffeat*2));
feat_2=feat_matrix(:,class_2_ind(1:nooffeat));
feat_3=feat_matrix(:,class_3_ind(1:nooffeat));

% feat_matrix1=[feat_1;feat_2;feat_3];
feat_matrix1=[feat_1,feat_2,feat_3];
% class_grnd_trth1=[class_1;class_2;class_3];
class_grnd_trth1=[class_1;class_2;class_3];

% class_grnd_trth1=class_grnd_trth1';% check here
feat_matrix1=feat_matrix1';
class_data_1=table(feat_matrix1,class_grnd_trth1);
%%
save('feat_flt_1.mat','class_data_1');