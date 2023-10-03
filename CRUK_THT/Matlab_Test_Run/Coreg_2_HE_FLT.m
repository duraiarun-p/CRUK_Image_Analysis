clc;clear;close all;
%%
base_dir_all='/mnt/local_share/TMA/FS-FLIM/raw/Tumour_2B/';
base_dir_all_lis=dir(base_dir_all);
base_dir_all_lis(1:2,:)=[];
% core_op_folder=[base_dir_all_lis(1).folder,'/',base_dir_all_lis(1).name,'/Mat_output2/'];
% core_stitched_masked_file=[core_op_folder,'core_stitched_masked_TX.mat'];


% for core_i =1 :length(base_dir_all_lis)

for core_i =1:1
    base_dir=[base_dir_all_lis(core_i).folder,'/',base_dir_all_lis(core_i).name,'/Mat_output2/'];
    core_stitched_masked_file=[base_dir,'core_stitched_masked_TX.mat'];
    if isfile(core_stitched_masked_file)
        disp('True')
cd(base_dir)
load("core_stitched_masked_TX.mat")
files=dir(base_dir);
files_names=cell(length(files),1);
for ind = 1 :length(files)
files_names{ind,1}=files(ind).name;
end
file_index_H=find(contains(files_names,'Row'));

hist_img=imread(files(file_index_H).name);
hist_img_siz=size(hist_img);
hist_img_gray=rgb2gray(hist_img);
hist_img_gray_mask=hist_img_gray;
hist_img_gray_mask(hist_img_gray_mask>200)=0;
hist_img_gray_mask(hist_img_gray_mask>0)=1;
hist_img_gray_mask_blurred=imgaussfilt(hist_img_gray_mask,20);
hist_mask=imfill(edge(hist_img_gray_mask_blurred),'holes');
% stitch_intensity_masked=permute(stitch_intensity_masked,[2,1]);
%Registration

fixed=stitch_intensity_masked;
fixed_siz=size(fixed);
movingRegistered_rgb=uint8(zeros([fixed_siz,3]));
moving=immultiply(hist_img(:,:,2),hist_mask);
moving_resized=imresize(moving,fixed_siz);

% [moving_gx,moving_gy]=imgradientxy(moving);

[optimizer,metric] = imregconfig("multimodal");
optimizer.InitialRadius = 0.009;
optimizer.Epsilon = 1.5e-4;
optimizer.GrowthFactor = 1.01;
optimizer.MaximumIterations = 10000;
%%
movingRegistered_rgb=uint8(zeros([fixed_siz,3]));
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
figure(1);imshowpair(fixed,movingRegistered)
figure(2),imshow(fixed,[]);title('Fixed Int')
figure(3),imshow(movingRegistered,[]);title('Registered')
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
imshowpair(stitch_intensity_masked,movingRegistered_rgb)

    else
        disp('False')
    end
end