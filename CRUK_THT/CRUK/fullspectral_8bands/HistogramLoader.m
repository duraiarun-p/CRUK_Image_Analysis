f = load('D:\Cancer_Full_Spectral\2_5_Alpha\Histograms\610nm\r2_c1_Histogram.fig','-mat');
axesNum = 1;
seriesNum = 1;
series = f.hgS_070000.children(axesNum).children(seriesNum);
NormalBins = series.properties.BinEdges;
NormalBinCounts = series.properties.BinCounts;


g = load('D:\Cancer_Full_Spectral\2_5_Alpha\Histograms\610nm\r2_c10_Histogram.fig','-mat');
axesNum = 1;
seriesNum = 1;
series = g.hgS_070000.children(axesNum).children(seriesNum);
CancerBins = series.properties.BinEdges;
CancerBinCounts = series.properties.BinCounts;


close all

figure('Renderer', 'painters', 'Position', [100 100 1500 610])
hold on
edgeNumber=size(NormalBins,2)-1;
meanSize = 20;
CancerBinCounts = movmean(CancerBinCounts,meanSize);
NormalBinCounts = movmean(NormalBinCounts,meanSize);

area(CancerBins(1:edgeNumber), CancerBinCounts/max(CancerBinCounts), 'FaceAlpha',0.5)
area(NormalBins(1:edgeNumber), NormalBinCounts/max(NormalBinCounts), 'FaceAlpha',0.5)

xlim([1 3])
legend( 'Cancer', 'Transitionary')
title('Lifetime Histogram at 605 nm')
xlabel('Lifetime (ns)')
ylabelPosition = ylabel('Normalised count');
ylabelPosition.Position(1) = ylabelPosition.Position(1) - 0.15;
ylabelPosition.Position(2) = ylabelPosition.Position(2) - 0.15;
hold off
set(findall(gcf,'-property','FontSize'),'FontSize',55)
%%
figure
difference = NormalBinCounts-CancerBinCounts;
plot(NormalBins, difference);
xlim([500 700])
legend('Transitionary Vs Cancer 500 nm')

% figure
% hold on
% plot(PlaqueSpecX, PlaqueSpecY/max(PlaqueSpecY))
% plot(PlaqueSpecX, HaloSpecY/max(HaloSpecY))
% %ylim([0 5])
% legend('Plaque', 'Halo')
% hold off
% 
% figure
% differecespectra = (PlaqueSpecY/max(PlaqueSpecY)) - (PlaqueSpecX/max(PlaqueSpecX));
% plot(PlaqueSpecX, differecespectra);
%%
close all
figure
shortWavelengthLifetime = squeeze(imageLifetimeData(20,:,:));
longWavelengthLifetime = squeeze(imageLifetimeData(250,:,:));
imagesc(shortWavelengthLifetime)
title('"Donor"')
figure
imagesc(longWavelengthLifetime)
title('"Acceptor"')
figure
imagesc(shortWavelengthLifetime - longWavelengthLifetime)
caxis([-1 3])
title('"Donor" - "Acceptor"')
colorbar
% Itensity
figure
shortWavelengthIntensity = squeeze(imageAlphaData(20,:,:));
longWavelengthLifetime = squeeze(imageAlphaData(200,:,:));
imagesc(shortWavelengthIntensity)
title('"Donor Intenisty"')
figure
imagesc(longWavelengthLifetime)
title('"Acceptor Intenisty"')
figure
imagesc(shortWavelengthIntensity - longWavelengthLifetime)
title('"Donor" - "Acceptor Intensity"')


%%
close all
spectralPixel = 20;
Td = imageLifetimeData(20,211,131);

FretEfficiency = 1-imageLifetimeData(spectralPixel,15:250,1:236)./Td;
imagesc(squeeze(FretEfficiency));
shading interp
colorbar