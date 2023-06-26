% numberofRows=3;
% numberofColums=3;
% numberofNondataFolders=3;
% row=0;
% % for r = 1:numberofRows
% %     row = row + 1
% % 
% %     colum = 0;
% %     for k = 1:numberofColums
% %         colum = colum + 1
% % %         disp('row')
% % %         disp(row)
% % %         disp('column')
% % %         disp(colum)
% %        
% % %         fileNumber = row+colum-1 + ((row-1)*(numberofColums-1))+numberofNondataFolders
% % %         imageNumber = row+colum-1 + ((row-1)*(numberofColums-1))
% % 
% % %         currDir = [filePath,'/',all_files(fileNumber).name]
% % %         disp(currDir)
% % %         cd(currDir)
% % %         disp('Loading workspace for folder:')
% % %         disp(all_files(fileNumber).name)
% % 
% % 
% %     end
% % end
% ind=1;
% for r=1:3
%    for c=1:3
%        row_ind(ind)=r;
%        col_ind(ind)=c;
%        ind=ind+1;
%    end
% end
%%
% clc;close all;clear;
% load('C:\Users\THT_CRUK\Documents\MATLAB\Test_Data\sample_analysis_load.mat');
% %%
% [SSIM_AB_C,SSIM_FB_C]=SSIM_Analysis(allIntensityImages,lifetimeAlphaData,lifetimeImageData);
% 
% function [SSIM_AB_C,SSIM_FB_C]=SSIM_Analysis(allIntensityImages,lifetimeAlphaData,lifetimeImageData)
% Tiles=size(allIntensityImages);
% SSIM_AB_C=cell(Tiles(2),1);
% SSIM_FB_C=cell(Tiles(2),1);
% for tile=1:Tiles(2)
%     I_B=squeeze(allIntensityImages{1,tile});
%     I_B_re=rescale(I_B);
%     I_AC=lifetimeAlphaData{1,tile};
%     siz_I_AC=size(I_AC);
%     SSIM_AB=zeros(siz_I_AC(1),1);
% 
%     I_FC=lifetimeImageData{1,tile};
%     siz_I_FC=size(I_FC);
%     SSIM_FB=zeros(siz_I_FC(1),1);
% 
%     for lambda=1:siz_I_AC(1)
%         I_A=squeeze(I_AC(lambda,:,:));
%         I_A_re=rescale(I_A);
%         SSIM_AB(lambda,1)=ssim(I_A_re,I_B_re);
%         
%         I_F=squeeze(I_FC(lambda,:,:));
%         I_F_re=rescale(I_F);
%         SSIM_FB(lambda,1)=ssim(I_F_re,single(I_B_re));
% 
%     end
%     SSIM_AB_C{tile,1}=SSIM_AB;
%     SSIM_FB_C{tile,1}=SSIM_FB;
% end
% end
%%
tic;
pause(0.25);
time_first=toc;
tic;
pause(0.5);
time_second=toc;



