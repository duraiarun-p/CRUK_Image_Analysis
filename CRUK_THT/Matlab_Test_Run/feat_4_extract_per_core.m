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
% for base_i = 5:5
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
stitch_intensity_cube_masked=double(rescale(stitch_intensity_cube_masked));
stitch_flt_cube_masked=double(stitch_flt_cube_masked);
% spec=38;
for spec=1:flt_siz(3)
slice_flt=stitch_flt_cube_masked(:,:,spec);
slice_int=stitch_intensity_cube_masked(:,:,spec);
% slice_flt_u8=uint8(255*mat2gray(slice_flt));
% slice_int_u8=uint8(255*mat2gray(slice_int));
slice_flt_u8=(1*mat2gray(slice_flt));
slice_int_u8=(1*mat2gray(slice_int));

% [new_slice, new_d]=imfuse(slice_flt_u8,slice_int_u8,'blend');
% flt_int(:,:,spec)=new_slice;% Intensity weighted lifetime
flt_int(:,:,spec)=slice_flt;%Lifetime
% flt_int(:,:,spec)=immultiply(slice_flt,slice_int);%Lifetime * Intensity
end

%%

figure(7),
imshow(HE)
%%
HE_siz=size(HE);
nbin=50;
no_of_param=4;
nopcs=50;
new_cell_width=20;
% cell_ind=1;
no_of_classes=3;
noofcells=length(bound_txed);
% feat_matrix=zeros(nbin*flt_siz(3),noofcells);
% feat_matrix=zeros(flt_siz(3)*nbin,noofcells);
feat_matrix=zeros(no_of_param*nbin,noofcells);
% feat_matrix=zeros(no_of_param,noofcells);
cells_img=cell(noofcells,2);
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
% cell_box=reshape(imresize3(cell_box,[new_cell_width,new_cell_width,flt_siz(3)]),new_cell_width*new_cell_width,flt_siz(3));
% cell_size=size(cell_box);
% cells_img{cell_ind,1}=cell_box(:);
% % cells_img{cell_ind,2}=class_grnd_trth(cell_ind);
% cells_img{cell_ind,2}=int2bit(class_grnd_trth(cell_ind),no_of_classes);
% % disp(cell_size);
% feat_matrix(:,cell_ind)=cell_box(:);

feat=zeros(nbin,flt_siz(3));
    for spec=1:flt_siz(3)
    [counts,flt_edge,flt_bins] = histcounts(cell_box(:,:,spec),nbin);
    feat(:,spec)=counts;
    end
    feat_1=zeros(nbin,no_of_param);
    for nbin_i=1:nbin
        feat_data=feat(nbin_i,:);
        pd=fitdist(feat_data','Normal');
        pd1 = fitdist(feat_data','Kernel','Kernel','epanechnikov');
        p=feat_data./sum(feat_data);
        p_log=log(p);
        p_log(p_log==-Inf)=0;
        H=-sum(p.*p_log);
        feat_1(nbin_i,:)=[pd.ParameterValues,pd1.Bandwidth,H];
    end

% [counts,flt_edge,flt_bins] = histcounts(cell_box(:),nbin);
% pd=fitdist(counts','Normal');
% p=counts./sum(counts);
% p_log=log(p);
% p_log(p_log==-Inf)=0;
% H=-sum(p.*p_log);
% feat_1=[pd.ParameterValues,H];
% feat_vector=rescale(feat(:));
% feat_vector=(feat(:));

% feat=zeros(nbin,flt_siz(3));
%     for spec=1:flt_siz(3)
%     % [counts,~] = imhist(cell_box(:,:,spec),nbin);
%     [counts,flt_edge,flt_bins] = histcounts(cell_box(:,:,spec),nbin);
%     feat(:,spec)=counts;
%     end
% feat_1=feat;

feat_vector=(feat_1(:));
feat_vector(isnan(feat_vector))=0;
feat_matrix(:,cell_ind)=feat_vector;
end

%%
% % dataSource = groundTruthDataSource(cells_img);
% grnd_truth=cell2table(cells_img);
% % grnd_truth(:,2)=class_grnd_trth;
% grnd_truth.Properties.VariableNames={'cells_img'  'labels'};
% 
% gnd_truth_ds=arrayDatastore(grnd_truth,"OutputType","same");
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
%%
% tot_x=floor([bound_txed(:,1);bound_txed(:,3)]);
% tot_y=floor([bound_txed(:,2);bound_txed(:,4)]);

% for ci=1:nooffeat
% bck_x = randi([1 flt_siz(1)],1,1);
% bck_y = randi([1 flt_siz(2)],1,1);
% if any(tot_x==bck_x) && any(tot_y==bck_y) ~= 0
%     bck_x = randi([1 flt_siz(1)],1,1);
%     bck_y = randi([1 flt_siz(2)],1,1);
% feat_4=zeros(nbin*flt_siz(3),nooffeat);
feat_4=zeros(nbin*no_of_param,nooffeat);
% feat_4=zeros(no_of_param,nooffeat);
class_4=zeros(nooffeat,1);



%%

% feat_matrix1=[feat_1;feat_2;feat_3;feat_4];
feat_matrix1=[feat_1,feat_2,feat_3,feat_4];
% class_grnd_trth1=[class_1;class_2;class_3];
class_grnd_trth1=[class_1;class_2;class_3;class_4];

% class_grnd_trth1=class_grnd_trth1';% check here
feat_matrix1=feat_matrix1';
class_data_1=table(feat_matrix1,class_grnd_trth1);

feat_matrix=feat_matrix';
class_data=table(feat_matrix,class_grnd_trth);
%%
feat_matrix2=[feat_1,feat_4];
% class_grnd_trth1=[class_1;class_2;class_3];
class_grnd_trth2=[class_1;class_4];
feat_matrix2=feat_matrix2';
class_data_2=table(feat_matrix2,class_grnd_trth2);
%% Save features as Mat file
save('feat_dist_1to4.mat','feat_matrix1','class_grnd_trth1');

%%
feat_matrix1=[feat_1,feat_2,feat_3];
% class_grnd_trth1=[class_1;class_2;class_3];
class_grnd_trth1=[class_1;class_2;class_3];

% class_grnd_trth1=class_grnd_trth1';% check here
feat_matrix1=feat_matrix1';
% class_data_1=table(feat_matrix1,class_grnd_trth1);
% save('feat_flt_2.mat','feat_matrix1','class_grnd_trth1');
%%
feat_matrix=feat_matrix';
% save('feat_flt_all.mat','feat_matrix','class_grnd_trth');
end