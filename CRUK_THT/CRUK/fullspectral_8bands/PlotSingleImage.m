
%% use this to plot a sinlge alpha adjusted image - the 1st value in the triples
% is the sensor pixel (convert to wavelength with lambda = pixel*0.549 +
% 500)

sensor_pixel = 20;
AlphaScaleFactor = 2.5;

%use one minus alpha = 1 to bring out low count regions
OneminusAlpha = 1;

if OneminusAlpha ==0
    
AlphatoPlot = squeeze(AlphaScaleFactor*(1-(lifetimeAlphaData(sensor_pixel,:,:)/max(max(lifetimeAlphaData(sensor_pixel,:,:))))));
    
else
AlphatoPlot = squeeze(AlphaScaleFactor*lifetimeAlphaData(sensor_pixel,:,:)/max(max(lifetimeAlphaData(sensor_pixel,:,:))));

end

 imagesc(squeeze(lifetimeImageData(sensor_pixel,:,:)), 'AlphaData', AlphatoPlot);
    set(gca,'Color','k')
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.3, 0.3, 0.3, 0.50]);
    ti = [ 0 0 0 0.09 ];
    set(gca,'Position',[ti(1) ti(2) 1-ti(3)-ti(1) 1-ti(4)-ti(2)]);
    set(gca,'units','centimeters')
    pos = get(gca,'Position');
    set(gcf, 'PaperUnits','centimeters');
    set(gcf, 'PaperSize', [pos(3)+ti(1)+ti(3) pos(4)+ti(2)+ti(4)]);
    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'PaperPosition',[0 0 pos(3)+ti(1)+ti(3) pos(4)+ti(2)+ti(4)]);
    set(gcf,'color','w');
    caxis([0.2 2])
    colorbar