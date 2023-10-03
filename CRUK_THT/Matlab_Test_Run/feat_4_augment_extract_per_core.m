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
nbin=50;
tumour_times=1;

class_T=class_grnd_trth(class_1_ind(1:nooffeat*tumour_times));
class_S=class_grnd_trth(class_2_ind(1:nooffeat));
class_I=class_grnd_trth(class_3_ind(1:nooffeat));

cells_T=cells_img(class_1_ind(1:nooffeat*tumour_times));
cells_S=cells_img(class_2_ind(1:nooffeat));
cells_I=cells_img(class_3_ind(1:nooffeat));

feat_matrix_S=zeros(flt_siz(3)*nbin,length(cells_S));
for cell_ind=1:length(cells_S)
cell_box_S=cells_S{cell_ind,1};
feat_S=zeros(nbin,flt_siz(3));
    for spec=1:flt_siz(3)
    % [counts,~] = imhist(cell_box(:,:,spec),nbin);
    [counts,flt_edge,flt_bins] = histcounts(cell_box_S(:,:,spec),nbin);
    feat_S(:,spec)=counts;
    end
% feat_vector=rescale(feat(:));
feat_vector_S=(feat_S(:));
feat_matrix_S(:,cell_ind)=feat_vector_S;
end

feat_matrix_I=zeros(flt_siz(3)*nbin,length(cells_I));
for cell_ind=1:length(cells_I)
cell_box_I=cells_I{cell_ind,1};
feat_I=zeros(nbin,flt_siz(3));
    for spec=1:flt_siz(3)
    % [counts,~] = imhist(cell_box(:,:,spec),nbin);
    [counts,flt_edge,flt_bins] = histcounts(cell_box_I(:,:,spec),nbin);
    feat_I(:,spec)=counts;
    end
% feat_vector=rescale(feat(:));
feat_vector_I=(feat_I(:));
feat_matrix_I(:,cell_ind)=feat_vector_I;
end

feat_matrix_T=zeros(flt_siz(3)*nbin,length(cells_I));
for cell_ind=1:length(cells_T)
cell_box_T=cells_T{cell_ind,1};
feat_T=zeros(nbin,flt_siz(3));
    for spec=1:flt_siz(3)
    % [counts,~] = imhist(cell_box(:,:,spec),nbin);
    [counts,flt_edge,flt_bins] = histcounts(cell_box_T(:,:,spec),nbin);
    feat_T(:,spec)=counts;
    end
% feat_vector=rescale(feat(:));
feat_vector_T=(feat_T(:));
feat_matrix_T(:,cell_ind)=feat_vector_T;
end


%% Data augmentation
augmenter = imageDataAugmenter('RandXReflection',1, 'RandYReflection',1,'RandRotation',[0 360]);

cells_I_aug=augment(augmenter,cells_I);
cells_S_aug=augment(augmenter,cells_S);

times_augment=4;
for time=1:times_augment
cells_I_aug=augment(augmenter,cells_I_aug);
cells_S_aug=augment(augmenter,cells_S_aug);
end

feat_matrix_S_aug=zeros(flt_siz(3)*nbin,length(cells_S));
for cell_ind=1:length(cells_S)
cell_box_S_aug=cells_S_aug{cell_ind,1};
feat_S_aug=zeros(nbin,flt_siz(3));
    for spec=1:flt_siz(3)
    % [counts,~] = imhist(cell_box(:,:,spec),nbin);
    [counts,flt_edge,flt_bins] = histcounts(cell_box_S_aug(:,:,spec),nbin);
    feat_S_aug(:,spec)=counts;
    end
% feat_vector=rescale(feat(:));
feat_vector_S_aug=(feat_S_aug(:));
feat_matrix_S_aug(:,cell_ind)=feat_vector_S_aug;
end

feat_matrix_I_aug=zeros(flt_siz(3)*nbin,length(cells_I));
for cell_ind=1:length(cells_I)
cell_box_I_aug=cells_I_aug{cell_ind,1};
feat_I_aug=zeros(nbin,flt_siz(3));
    for spec=1:flt_siz(3)
    % [counts,~] = imhist(cell_box(:,:,spec),nbin);
    [counts,flt_edge,flt_bins] = histcounts(cell_box_I_aug(:,:,spec),nbin);
    feat_I_aug(:,spec)=counts;
    end
% feat_vector=rescale(feat(:));
feat_vector_I_aug=(feat_I_aug(:));
feat_matrix_I_aug(:,cell_ind)=feat_vector_I_aug;
end
%%
% feat_matrix_S=[feat_matrix_S,feat_matrix_S];
% feat_matrix_I=[feat_matrix_I,feat_matrix_I];
% class_S=[class_S;class_S];
% class_I=[class_I;class_I];

feat_matrix_O=[feat_matrix_T,feat_matrix_S,feat_matrix_I];
class_O=[class_T;ones(length(class_S)*2,1)*5];



%%
% save('feat_23.mat','feat_matrix_S','class_S','feat_matrix_I','class_I');
save('feat_15.mat','feat_matrix_O','class_O');

end