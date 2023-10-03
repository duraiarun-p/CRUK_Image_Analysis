clc;clear;close all;
%%
% base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-2_20230215/Mat_output';
% base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-9_20230222/Mat_output';
% base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-13_20230226/Mat_output';
% base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-3_Col-5_20230218/Mat_output2';
% base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/RT/Row-4_Col-1_20230214/Mat_output2';
% base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/RT/Row-6_Col-10_20230223/Mat_output2';
%%
% base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/Row-1_Col-9_20230222/FLT_IMG_DIR/Stitched';

base_dir_all=cell(5,1);
base_dir_all{1,1}='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-2_20230215/Mat_output';
base_dir_all{2,1}='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-9_20230222/Mat_output';
base_dir_all{3,1}='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-13_20230226/Mat_output';
% base_dir_all='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-3_Col-5_20230218/Mat_output2';
base_dir_all{4,1}='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/RT/Row-4_Col-1_20230214/Mat_output2';
base_dir_all{5,1}='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/RT/Row-6_Col-10_20230223/Mat_output2';

% for base_i = 1:length(base_dir_all)
for base_i = 2:2
%% Load stitched and masked flt cubes
base_dir=base_dir_all{base_i,1};
% base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Mat_output2';
cd(base_dir)
load("core_stitched_masked.mat")
%%

files=dir(base_dir);
files_names=cell(length(files),1);
for ind = 1 :length(files)
files_names{ind,1}=files(ind).name;
end
file_index_H=find(contains(files_names,'Row'));
%% Load H&E image
hist_img=imread(files(file_index_H).name);
hist_img_siz=size(hist_img);
%%
% stitch_intensity_masked=permute(stitch_intensity_masked,[2,1]);
%% Registration

fixed=stitch_intensity_masked;
fixed_siz=size(fixed);
movingRegistered_rgb=uint8(zeros([fixed_siz,3]));
moving=hist_img(:,:,2);
moving_resized=imresize(moving,fixed_siz);
[optimizer,metric] = imregconfig("multimodal");
optimizer.InitialRadius = 0.009;
optimizer.Epsilon = 1.5e-4;
optimizer.GrowthFactor = 1.01;
optimizer.MaximumIterations = 10000;

% movingRegistered = imregister(moving_resized,fixed,"affine",optimizer,metric);
tform_HF = imregtform(moving_resized,fixed,"affine",optimizer,metric);
movingRegistered = imwarp(moving_resized,tform_HF,"OutputView",imref2d(size(fixed)));
for channel=1:3
    moving_resized_channel=imresize(hist_img(:,:,channel),fixed_siz);
    movingRegistered_rgb(:,:,channel)=imwarp(moving_resized_channel,tform_HF,"OutputView",imref2d(size(fixed)));
end
%% Visualisation
x=fixed_siz(1)/4;
y=fixed_siz(2)/4;
% x=floor(1.448300000000000e+03);
% y=floor(3.060200000000000e+03);
figure(5);
imshow(hist_img);
gca;
hold on
% [x,y]=ginput(1);
plot(x,y, 'r+', 'MarkerSize', 30, 'LineWidth', 2);
hold off;
title('get output');
figure(1);imshowpair(fixed,movingRegistered,"Scaling","joint")
figure(2),imshow(fixed,[]);title('Fixed Int')
figure(3),imshow(movingRegistered,[]);title('Registered G')
figure(4),imshow(movingRegistered_rgb);title('Registered RGB')
%% Transform calculation
r_old=x; c_old=y;
scale=fixed_siz./hist_img_siz(1:2);
a = [scale(2), 0, 0;
0, scale(1), 0;
0, 0, 1];
T_resize = affinetform2d(a);
[r_new,c_new] = transformPointsForward(T_resize,r_old,c_old);
[r_new1,c_new1] = transformPointsForward(tform_HF,r_new,c_new);
%% Post transform validation
figure(6),
imshow(movingRegistered_rgb);
axis on
hold on;
plot(r_new1,c_new1, 'r+', 'MarkerSize', 30, 'LineWidth', 2);
% plot(mid_r_new1,mid_c_new1, 'r+', 'MarkerSize', 30, 'LineWidth', 2);
hold off;
title('Registered RGB - with points')
figure(7),
imshowpair(stitch_intensity_masked,movingRegisteredRGB,"Scaling","joint")

%% Save registration results

% imwrite(movingRegistered_rgb,'coreg_HE.tiff');
% save('tforms.mat','tform_HF','T_resize');
disp(base_dir)
end