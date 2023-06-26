%%test analysis script
clc;clear;close all;

%%
 load('C:\Users\THT_CRUK\Documents\MATLAB\Test_Output\2.mat','allIntensityImages','allIntensityImages_FEXP','allIntensityImages_PEXP','lifetimeAlphaData','lifetimeAlphaData_PEXP','lifetimeAlphaData_FEXP','lifetimeImageData','lifetimeImageData_PEXP','lifetimeImageData_FEXP');
%%

MA=zeros(2,2);
SSIM=zeros(2,2);

for tile=1:2

LT_LS=lifetimeImageData{1,tile};
LT_PEXP=lifetimeImageData_PEXP{1,tile};
LT_FEXP=lifetimeImageData_FEXP{1,tile};
[MA_LT,SSIM_LT,MA_LS_PEXP,MA_LS_FEXP,SSIM_LS_PEXP,SSIM_LS_FEXP]=myimganalysis(LT_LS,LT_PEXP,LT_FEXP);

MA(tile,:)=MA_LT;
SSIM(tile,:)=SSIM_LT;

LT_alpha_LS=lifetimeAlphaData{1,tile};
LT_alpha_PEXP=lifetimeAlphaData_PEXP{1,tile};
LT_alpha_FEXP=lifetimeAlphaData_FEXP{1,tile};
[MA_LT_alpha,SSIM_LT_alpha,MA_LS_PEXP_alpha,MA_LS_FEXP_alpha,SSIM_LS_PEXP_alpha,SSIM_LS_FEXP_alpha]=myimganalysis(LT_alpha_LS,LT_alpha_PEXP,LT_alpha_FEXP);

LT_inte_LS=allIntensityImages{1,tile};
LT_inte_PEXP=allIntensityImages_PEXP{1,tile};
LT_inte_FEXP=allIntensityImages_FEXP{1,tile};
[MA_LT_inte,SSIM_LT_inte,MA_LS_PEXP_inte,MA_LS_FEXP_inte,SSIM_LS_PEXP_inte,SSIM_LS_FEXP_inte]=myimganalysis(LT_inte_LS,LT_inte_PEXP,LT_inte_FEXP);
end
%%
for tile=1:2

myplot(LT_inte_LS,LT_LS,LT_alpha_LS,100*tile);
myplot(LT_inte_PEXP,LT_PEXP,LT_alpha_PEXP,300*tile);
myplot(LT_inte_FEXP,LT_FEXP,LT_alpha_FEXP,500*tile)



end

function myplot(allIntensityImages,lifetimeImageData,lifetimeAlphaData,fignum)

   nspectrum_split=16;

  n_spectrum_siz = size(lifetimeImageData);
  n_spectrum=n_spectrum_siz(1);
 allIntensityImages=squeeze(allIntensityImages);

 plot_spectrum=1:nspectrum_split:n_spectrum;
 plotspectrumL=length(plot_spectrum);
 lifetimeImageData_selected=lifetimeImageData(plot_spectrum,:,:);

 lifetimeAlphaData_selected=lifetimeAlphaData(plot_spectrum,:,:);
 figure(fignum+1);
imshow(allIntensityImages,[]);colorbar;
title('allintensityImage');

plotrow=round(sqrt(plotspectrumL));
plotcol=round(plotspectrumL/plotrow);
clim_max=4;
 clim_min=1.0;
figure(fignum+2);
for plot_index=1:plotspectrumL
subplot(plotrow,plotcol,plot_index)
imagesc(squeeze(lifetimeImageData_selected(plot_index,:,:)));
clim([clim_min clim_max])
set(gca,'Color','k');
colormap jet;
subtitle(['\lambda_{',num2str(plot_spectrum(plot_index)),'}']);
end

 clim_max1=max(lifetimeAlphaData_selected(:));
 clim_min1=min(lifetimeAlphaData_selected(:));
figure(fignum+3);
for plot_index=1:plotspectrumL
subplot(plotrow,plotcol,plot_index)
imagesc(squeeze(lifetimeAlphaData_selected(plot_index,:,:)));
clim([clim_min1 clim_max1])
set(gca,'Color','k');
colormap jet;
subtitle(['\lambda_{',num2str(plot_spectrum(plot_index)),'}']);
end

clim_min1=1;
clim_max2=4;
figure(fignum+3);
for plot_index=1:plotspectrumL
subplot(plotrow,plotcol,plot_index)
h=imagesc(squeeze(lifetimeImageData_selected(plot_index,:,:)));
clim([clim_min1 clim_max2])
I=squeeze(lifetimeAlphaData_selected(plot_index,:,:));
In=I/max(I(:));
In=In*2.5;
set(h, 'AlphaData', In);
set(gca,'Color','k');
colormap jet;
subtitle(['\lambda_{',num2str(plot_spectrum(plot_index)),'}']);
end



end

function [MA_LT,SSIM_LT,MA_LS_PEXP,MA_LS_FEXP,SSIM_LS_PEXP,SSIM_LS_FEXP]=myimganalysis(LT_LS,LT_PEXP,LT_FEXP)

sizLF=size(LT_LS);
lamdas=sizLF(1);

MA_LT_E1=LT_LS-LT_PEXP;
MA_LT(1)=mean(MA_LT_E1(:));
MA_LT_E2=LT_LS-LT_FEXP;
MA_LT(2)=mean(MA_LT_E2(:));

MA_LS_PEXP=myerrorsum(MA_LT_E1,lamdas);
MA_LS_FEXP=myerrorsum(MA_LT_E2,lamdas);

% sizLF=size(LT_LS);
% lamdas=sizLF(1);
SSIM_LS_PEXP=myssim(LT_LS,LT_PEXP,lamdas);
SSIM_LS_FEXP=myssim(LT_LS,LT_FEXP,lamdas);

SSIM_LT(1)=mean(SSIM_LS_PEXP);
SSIM_LT(2)=mean(SSIM_LS_FEXP);
end

function MA=myerrorsum(MA_LT_E1,lamdas)
MA=zeros(lamdas,1);
for lambda=1:lamdas
        I_A=squeeze(MA_LT_E1(lambda,:,:));
        MA=sum(I_A);
end
end

function SSIM_FB=myssim(I_AC,I_FC,lamdas)
SSIM_FB=zeros(lamdas,1);
    for lambda=1:lamdas
        I_A=squeeze(I_AC(lambda,:,:));
        I_A_re=rescale(I_A);
%         SSIM_AB(lambda,1)=ssim(I_A_re,I_B_re);
        
        I_F=squeeze(I_FC(lambda,:,:));
        I_F_re=rescale(I_F);
%         SSIM_FB(lambda,1)=ssim(I_A_re,single(I_F_re));
    SSIM_FB(lambda,1)=ssim(I_A_re,I_F_re);

    end
end