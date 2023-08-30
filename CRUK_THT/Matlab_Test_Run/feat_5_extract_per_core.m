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

% for base_i = 1:length(base_dir_all)
for base_i = 1:1
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

% u8_flt=uint8(zeros(size(stitch_flt_cube_masked)));
% u8_int=uint8(zeros(size(stitch_intensity_cube_masked)));
flt_int=double(zeros(size(stitch_intensity_cube_masked)));
flt_siz=size(stitch_intensity_cube_masked);
% stitch_intensity_cube_masked=double(rescale(stitch_intensity_cube_masked));
% stitch_flt_cube_masked=double(stitch_flt_cube_masked);
% spec=38;
% for spec=1:flt_siz(3)
% slice_flt=stitch_flt_cube_masked(:,:,spec);
% slice_int=stitch_intensity_cube_masked(:,:,spec);
% % slice_flt_u8=uint8(255*mat2gray(slice_flt));
% % slice_int_u8=uint8(255*mat2gray(slice_int));
% slice_flt_u8=(1*mat2gray(slice_flt));
% slice_int_u8=(1*mat2gray(slice_int));
% 
% % [new_slice, new_d]=imfuse(slice_flt_u8,slice_int_u8,'blend');
% % flt_int(:,:,spec)=new_slice;% Intensity weighted lifetime
% flt_int(:,:,spec)=slice_flt;%Lifetime
% % flt_int(:,:,spec)=immultiply(slice_flt,slice_int);%Lifetime * Intensity
% end

%%

figure(7),
imshow(HE)
%%
nbin=20;
no_of_param=2;
nopcs=50;
new_cell_width=20;
cell_ind=1;
no_of_classes=3;
noofcells=length(bound_txed);
% feat_matrix=zeros(nbin*flt_siz(3),noofcells);
% feat_matrix=zeros(flt_siz(3)*nbin,noofcells);
% feat_matrix=zeros(no_of_param*nbin,noofcells);
% feat_matrix=zeros(no_of_param,noofcells);
feat_matrix=zeros(noofcells,no_of_param);
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

cell_box=stitch_flt_cube_masked(row_start:row_stop,col_start:col_stop,:);
cell_box1=stitch_intensity_cube_masked(row_start:row_stop,col_start:col_stop,:);
cell_box_HE=HE(row_start:row_stop,col_start:col_stop,:);

feat_2=[sum(cell_box1(:)),mean(cell_box(:))];
feat_matrix(cell_ind,:)=feat_2;
end

class_1_ind=find(class_grnd_trth==1);
class_2_ind=find(class_grnd_trth==2);
class_3_ind=find(class_grnd_trth==3);

class_1=class_grnd_trth(class_1_ind);
class_2=class_grnd_trth(class_2_ind);
class_3=class_grnd_trth(class_3_ind);

% feat_1=feat_matrix(class_1_ind(1:1000),:);
% feat_2=feat_matrix(class_2_ind(1:1000),:);
% feat_3=feat_matrix(class_3_ind(1:1000),:);

feat_1=feat_matrix(class_1_ind,:);
feat_2=feat_matrix(class_2_ind,:);
feat_3=feat_matrix(class_3_ind,:);

figure(1);
plot(feat_1(:,2),feat_1(:,1),'.');hold on;
plot(feat_2(:,2),feat_2(:,1),'.');
plot(feat_3(:,2),feat_3(:,1),'.');hold off;
legend ('tumour','stroma','immune');
title('Lifetime vs Intensity');
figure(2);plot(feat_1(:,1),'.');hold on;
plot(feat_2(:,1),'.');plot(feat_3(:,1),'.');hold off;
legend ('tumour','stroma','immune');
title('Intensity');
figure(3);plot(feat_1(:,2),'.');hold on;
plot(feat_2(:,2),'.');plot(feat_3(:,2),'.');hold off;
legend ('tumour','stroma','immune');
title('Lifetime');
end