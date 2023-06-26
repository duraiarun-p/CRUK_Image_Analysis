%% Analysis of LFS and SEXP Fitting
clc;clear;
%close all;
%%
load('C:\Users\THT_CRUK\Documents\MATLAB\Test_Data\sample_analysis_load_512.mat');
[SSIM_F12,IMDIFF_F12]=compare_lfs_exp(allIntensityImages1,lifetimeAlphaData1,lifetimeImageData1,lifetimeImageData2,10,'LS vs Partial Exp');
[SSIM_F13,IMDIFF_F13]=compare_lfs_exp(allIntensityImages1,lifetimeAlphaData1,lifetimeImageData1,lifetimeImageData3,20,'LS vs Full Exp');



%%


function [SSIM_F,IMDIFF_F]=compare_lfs_exp(allIntensityImages1,lifetimeAlphaData1,lifetimeImageData1,lifetimeImageData2,fignum,titlestring)

Tiles=size(allIntensityImages1);
sizLF=size(lifetimeAlphaData1{1,end});
SSIM_F=cell(Tiles(2),4);
IMDIFF_F=cell(Tiles(2),4);
plotrow=ceil(sqrt(Tiles(2)));

for tile=1:Tiles(2)

I_FC1=lifetimeImageData1{1,tile};
siz_I_FC=size(I_FC1);
lambdas=sizLF(1);

I_FC2=lifetimeImageData2{1,tile};
SSIM_F12=zeros(siz_I_FC(1),1);

IF12=zeros(siz_I_FC);
IF12D=zeros(siz_I_FC);
IEF12=zeros(siz_I_FC(1),1);
SSIM_F12_Map=zeros(siz_I_FC);

IF12D=zeros(siz_I_FC);
IEF12D=zeros(siz_I_FC(1),1);

    for lambda=1:lambdas
            I_F1=squeeze(I_FC1(lambda,:,:));
            I_F_re1=rescale(I_F1);
            
            I_F2=squeeze(I_FC2(lambda,:,:));
            I_F_re2=rescale(I_F2);
    
            I_F_12=imabsdiff(I_F1,I_F2);
            IF12(lambda,:,:)=I_F_12;


%             I_F_12_D=I_F1-I_F2;
            I_F_12_D=I_F2-I_F1;
            IF12D(lambda,:,:)=I_F_12_D;
            IEF12D(lambda,1)=sum(I_F_12_D(:));


%             IEF12=immse(I_F1,I_F2);
            IEF12=immse(I_F_re1,I_F_re2);
    
            [SSIM_F12(lambda,1),SSIM_F12_Map_C]=ssim(I_F_re2,I_F_re1);
    
            SSIM_F12_Map(lambda,:,:)=SSIM_F12_Map_C;
    
    %         figure(1),
    %         subplot(3,3,lambda),
    %         imshow(SSIM_F12_Map_C,[]);
    %         
    %         figure(2),
    %         subplot(3,3,lambda),
    %         imshow(I_F_12,[]);
    %         pause(0.2);
    
    end

    SSIM_F{tile,1}=SSIM_F12_Map;
    SSIM_F{tile,2}=squeeze(sum(SSIM_F12_Map,1));
    SSIM_F{tile,3}=SSIM_F12;
    SSIM_F{tile,4}=mean(SSIM_F12);

    IMDIFF_F{tile,1}=IF12;
    IMDIFF_F{tile,2}=squeeze(sum(IF12,1));
    IMDIFF_F{tile,3}=sum(IEF12D);
    IMDIFF_F{tile,4}=IEF12D;

    figure(fignum),
%     title(titlestring),
    subplot(plotrow,plotrow,tile),
    imagesc(IMDIFF_F{tile,2}),colormap(hot);
    subtitle(['ME = ',num2str(IMDIFF_F{tile,3})])

    figure(fignum+1),
%     title(titlestring),
    subplot(plotrow,plotrow,tile),
    imagesc(SSIM_F{tile,2}),colormap(hot);
    subtitle(['SSIM = ',num2str(SSIM_F{tile,4})])

end


end