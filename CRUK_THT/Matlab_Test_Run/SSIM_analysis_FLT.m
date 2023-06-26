clc;clear;close all;
%%
load('C:\Users\THT_CRUK\Documents\MATLAB\Test_Data\sample_analysis_load_512.mat');
%%
[SSIM_AB_C1,SSIM_FB_C1,SSIM_AB_D1,SSIM_FB_D1]=SSIM_Analysis(allIntensityImages1,lifetimeAlphaData1,lifetimeImageData1);
[SSIM_AB_C2,SSIM_FB_C2,SSIM_AB_D2,SSIM_FB_D2]=SSIM_Analysis(allIntensityImages2,lifetimeAlphaData2,lifetimeImageData2);
[SSIM_AB_C3,SSIM_FB_C3,SSIM_AB_D3,SSIM_FB_D3]=SSIM_Analysis(allIntensityImages3,lifetimeAlphaData3,lifetimeImageData3);
%%
% SSIM_AB_C1_D=cell2mat(SSIM_AB_C1);
% SSIM_AB_C1_D=unique(SSIM_AB_C1_D);
%%

function [SSIM_AB_C,SSIM_FB_C,SSIM_AB_D,SSIM_FB_D]=SSIM_Analysis(allIntensityImages,lifetimeAlphaData,lifetimeImageData)
Tiles=size(allIntensityImages);
SSIM_AB_C=cell(Tiles(2),1);
SSIM_FB_C=cell(Tiles(2),1);
sizLF=size(lifetimeAlphaData{1,16});
lamdas=sizLF(1);
SSIM_AB_D=zeros(Tiles(2),lamdas);
SSIM_FB_D=zeros(Tiles(2),lamdas);

for tile=1:Tiles(2)
    I_B=squeeze(allIntensityImages{1,tile});
    I_B_re=rescale(I_B);
    I_AC=lifetimeAlphaData{1,tile};
    siz_I_AC=size(I_AC);
    SSIM_AB=zeros(siz_I_AC(1),1);

    I_FC=lifetimeImageData{1,tile};
    siz_I_FC=size(I_FC);
    SSIM_FB=zeros(siz_I_FC(1),1);

    for lambda=1:lamdas
        I_A=squeeze(I_AC(lambda,:,:));
        I_A_re=rescale(I_A);
        SSIM_AB(lambda,1)=ssim(I_A_re,I_B_re);
        
        I_F=squeeze(I_FC(lambda,:,:));
        I_F_re=rescale(I_F);
        SSIM_FB(lambda,1)=ssim(I_F_re,single(I_B_re));

    end
    SSIM_AB_C{tile,1}=SSIM_AB;
    SSIM_FB_C{tile,1}=SSIM_FB;

    SSIM_AB_D(tile,:)=SSIM_AB;
    SSIM_FB_D(tile,:)=SSIM_FB;
end
end