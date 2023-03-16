%% Load 2 MEAN TAU figures and plot both and the difference
f = load('C:\Proteus\Data\20201016_161020Susan\SF161020\Histograms\meanTau\r1_c1_MeanTau.fig','-mat');
axesNum = 2;
seriesNum = 1;
series = f.hgS_070000.children(axesNum).children(seriesNum);
NormalX = series.properties.XData;
NormalY = series.properties.YData;


g = load('C:\Proteus\Data\20201016_161020Susan\SF161020\Histograms\meanTau\r3_c1_MeanTau.fig','-mat');
axesNum = 2;
seriesNum = 1;
series = g.hgS_070000.children(axesNum).children(seriesNum);
CancerX = series.properties.XData;
CancerY = series.properties.YData;

%
close all
%

firstPixel = 10;
lastPixel = 310

figure('Renderer', 'painters', 'Position', [100 100 1500 610])
hold on
plot(CancerX(firstPixel:lastPixel), CancerY(firstPixel:lastPixel), 'LineWidth' , 4)
plot(NormalX(firstPixel:lastPixel), NormalY(firstPixel:lastPixel), 'LineWidth' , 4)
ylim([-1 4])
%xlim([500 640])
%yticks([0 2 2.5 3 5])

difference = NormalY(firstPixel:lastPixel)-CancerY(firstPixel:lastPixel);
%plot(NormalX(1:360), difference(1:360)./max(difference(1:360)));
plot(NormalX(firstPixel:lastPixel), difference, 'LineWidth' , 4);
title('Spectral Lifetime of Unstained Tissue')
ylabel('lifetime (ns)')
xlabel('Wavelength (nm)')
lgd =legend('Cancer' ,'Normal', 'Difference');

set(findall(gcf,'-property','FontSize'),'FontSize',55)
lgd.FontSize = 40;
hold off
xticks([0 1 2 3])
%
disp('Mean of 1st loaded figure and SD')
meanDifference = mean(NormalY(firstPixel:lastPixel))
StandardDeviationoftheDifference = std(NormalY(firstPixel:lastPixel))

disp('Mean of 2nd loaded figure and SD')
meanDifference = mean(CancerY(firstPixel:lastPixel))
StandardDeviationoftheDifference = std(CancerY(firstPixel:lastPixel))


disp('Mean of Difference and SD')
meanDifference = mean(difference)
StandardDeviationoftheDifference = std(difference)
%% PLOT ONLY THE MEAN TAU
figure('Renderer', 'painters', 'Position', [100 100 1500 610])
hold on
plot(NormalX(firstPixel:lastPixel), NormalY(firstPixel:lastPixel), 'LineWidth' , 5)
plot(CancerX(firstPixel:lastPixel), CancerY(firstPixel:lastPixel), 'LineWidth' , 5)
ylim([0 2.8])
xlim([505 750])

title('Spectral lifetime of unstained tissue')
ylabel('lifetime (ns)')

lgd =legend('Transitionary', 'Cancer');

set(findall(gcf,'-property','FontSize'),'FontSize',55)
lgd.FontSize = 40;
hold off

% To turn X axis on / off, swap the commenting of the next 2 lines
%xticks([])
xlabel('Wavelength (nm)')

%% PLOT ONLY THE DIFFERENCE
figure('Renderer', 'painters', 'Position', [100 100 1500 500])
difference = NormalY-CancerY;
%plot(NormalX(1:360), difference(1:360)./max(difference(1:360)));
plot(NormalX(1:400), difference(1:400), 'LineWidth' , 4, 'Color', 'r');
ylim([-0.1 1])
xlim([500 689])
xlabel('Wavelength (nm)')
lgd =legend('Difference');
yticks([0 0.4])
set(findall(gcf,'-property','FontSize'),'FontSize',55)
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