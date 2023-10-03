 clc;clear;close all;
 %%
base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-9_20230222/Mat_output';
% base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-2_20230215/Mat_output';
cd(base_dir)
files=dir(base_dir);
files_names=cell(length(files),1);
for ind = 1 :length(files)
files_names{ind,1}=files(ind).name;
end
file_index_H=find(contains(files_names,'Row'));
file_index_R=find(contains(files_names,'hist'));
load("warp_matrix.mat")
%%
hist_img=imread(files(file_index_H).name);
hist_img_R=imread(files(file_index_R).name);

hist_img_R_siz=size(hist_img_R);
hist_img_siz=size(hist_img);

%%
hist_img_R_mat=imresize(hist_img,[hist_img_R_siz(1),hist_img_R_siz(2)]);
scale=hist_img_R_siz./hist_img_siz;
% hist_img_R_mat=imresize(hist_img,[hist_img_R_siz(1),hist_img_siz(2)]);
% hist_img_R_mat=imresize(hist_img,0.5);
% hist_img_R_siz=hist_img_siz/2;

% x=hist_img_siz(1)/2;
% y=hist_img_siz(2)/2;
%%
figure(1),
imshow(hist_img);
hold on
[x,y]=ginput(1);
% plot(x,y, 'r+', 'MarkerSize', 30, 'LineWidth', 2);
hold off;

%%

mid_r_old=hist_img_siz(1)/2;
mid_c_old=hist_img_siz(2)/2;

r_old=x;
c_old=y;


hist_img_rows=1:hist_img_siz(1);hist_img_rows=hist_img_rows';
hist_img_cols=1:hist_img_siz(2);hist_img_cols=hist_img_cols';

[hist_img_rows_R,map1]=imresize(hist_img_rows,[hist_img_R_siz(1),1]);
[hist_img_cols_R,map2]=imresize(hist_img_cols,[hist_img_R_siz(2),1]);

hist_img_rows_R1=1:hist_img_R_siz(1);hist_img_rows_R1=hist_img_rows_R1';
hist_img_cols_R1=1:hist_img_R_siz(2);hist_img_cols_R1=hist_img_cols_R1';

% r_new = interp1(hist_img_rows_R, hist_img_rows_R, r_old);
% c_new = interp1(hist_img_cols_R, hist_img_cols_R, c_old);

% % r_new = interp1([1 hist_img_siz(1)/2 hist_img_siz(1)], [1 hist_img_R_siz(1)/2 hist_img_R_siz(1)], r_old,'linear');
% % c_new = interp1([1 hist_img_siz(2)/2 hist_img_siz(2)], [1 hist_img_R_siz(2)/2 hist_img_R_siz(2)], c_old,'linear');

% r_new = interp1([1 hist_img_siz(1)], [1 hist_img_R_siz(1)], r_old);
% c_new = interp1([1 hist_img_siz(2)], [1 hist_img_R_siz(2)], c_old);

% 
% r_new = interp1([hist_img_rows(1) hist_img_rows(end)], [hist_img_rows_R(1) hist_img_rows_R(end)], r_old);
% c_new = interp1([hist_img_cols(1) hist_img_cols(end)], [hist_img_cols_R(1) hist_img_cols_R(end)], c_old);

% ratio_row=hist_img_R_siz(1)/hist_img_siz(1);
% ratio_col=hist_img_R_siz(2)/hist_img_siz(2);

% r_new=(r_old-(hist_img_siz(1)/2))*ratio_row+(hist_img_rows_R(1)/2);
% c_new=(c_old-(hist_img_siz(2)/2))*ratio_col+(hist_img_rows_R(2)/2);

% r_new=r_old*ratio_row;
% c_new=c_old*ratio_col;

% r_new=r_old*scale(1);
% c_new=c_old*scale(2);

% r_new=((hist_img_siz(1)/2)-r_old)*scale(1); 
% c_new=((hist_img_siz(2)/2)-c_old)*scale(2);

% p1   =  0.4701  ; 
% p2   =  0.2651  ;
% 
% r_new=r_old*p1+p2;
% 
% p1  =  0.4990  ;
% p2  = 0.2506  ;
% 
% c_new=c_old*p1+p2;
sc=scale;
a = [sc(2), 0, 0;
0, sc(1), 0;
0, 0, 1];

T = affinetform2d(a);
[r_new,c_new] = transformPointsForward(T,r_old,c_old);
[mid_r_new,mid_c_new] = transformPointsForward(T,mid_r_old,mid_c_old);
%%
figure(2),
imshow(hist_img);
axis on
hold on;
plot(r_old,c_old, 'r+', 'MarkerSize', 30, 'LineWidth', 2);
plot(mid_r_old,mid_c_old, 'r+', 'MarkerSize', 30, 'LineWidth', 2);
hold off
%%
figure(3),
imshow(hist_img_R_mat);
axis on
hold on;
plot(r_new,c_new, 'r+', 'MarkerSize', 30, 'LineWidth', 2);
plot(mid_r_new,mid_c_new, 'r+', 'MarkerSize', 30, 'LineWidth', 2);
hold off;
%%
warp_matrix1=warp_matrix;
% warp_matrix1=flipud(warp_matrix);
% warp_matrix1=transpose(warp_matrix);
% warp_matrix1=zeros(3,3);
% warp_matrix1(1:2,1:3)=warp_matrix;
% warp_matrix1(3,:)=[0 0 1];
% warp_matrix1=zeros(size(warp_matrix));
% warp_matrix1(:,2)=warp_matrix(:,1);
% warp_matrix1(:,1)=warp_matrix(:,2);
T1 = affinetform2d(warp_matrix1);
[r_new1,c_new1] = transformPointsForward(T1,r_new,c_new);
[mid_r_new1,mid_c_new1] = transformPointsForward(T1,mid_r_new,mid_c_new);
%%
figure(4),
imshow(hist_img_R);
axis on
hold on;
plot(r_new1,c_new1, 'r+', 'MarkerSize', 30, 'LineWidth', 2);
plot(mid_r_new1,mid_c_new1, 'r+', 'MarkerSize', 30, 'LineWidth', 2);
hold off;
figure(5),
imshow(hist_img_R);
hold on;
plot(r_new1,c_new1, 'r+', 'MarkerSize', 30, 'LineWidth', 2);
plot(mid_r_new1,mid_c_new1, 'r+', 'MarkerSize', 30, 'LineWidth', 2);
hold off;