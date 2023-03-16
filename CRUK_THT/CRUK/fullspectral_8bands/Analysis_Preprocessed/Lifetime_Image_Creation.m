function Lifetime_Image_Creation(spectral_pixel, lifetimeDatatoAnalyse, alphaDatatoAnalyse,  wavelengths, lifetimeLow, lifetimeHigh, filePath, row, colum,analysisType, AlphaScalefactor, oneMinusAlpha)
    CurrentWavelength = round(spectral_pixel*0.5468 + 500);
    
    
    % Plot lifetime image
    caption = [num2str(CurrentWavelength), 'nm'];
    close all
    image = squeeze(lifetimeDatatoAnalyse(spectral_pixel,:,:));
    histogram(image, 1000)
    xlim([lifetimeLow lifetimeHigh+1])
    xlabel('Lifetime (ns)')
    ylabel('Counts')
    set(gcf,'color','w');
    title(caption);
    imagewd = getframe(gcf); 
    imwrite(imagewd.cdata, [filePath,'\New Analysis',analysisType, '\Histograms\', num2str(CurrentWavelength),'nm','\r', num2str(row),'_c', num2str(colum), '_Histogram.tif']);
    savefig([filePath,'\New Analysis',analysisType,  '\Histograms\', num2str(CurrentWavelength),'nm','\r', num2str(row),'_c', num2str(colum), '_Histogram.fig']);
    save([filePath,'\New Analysis', analysisType, '\Histograms\', num2str(CurrentWavelength),'nm','\r', num2str(row),'_c', num2str(colum), '_Histogram.mat'],'image')
    close(gcf);

    figure
    AlphaData = squeeze(alphaDatatoAnalyse(spectral_pixel,:,:));
    if oneMinusAlpha == 1
        bins_array_alpha_normalised = AlphaScalefactor*(1-AlphaData/max(max(AlphaData)));
    % try scaling / 1-alpha
    else
        bins_array_alpha_normalised = AlphaScalefactor*AlphaData/max(max(AlphaData));
    end

    imagesc(image, 'AlphaData', bins_array_alpha_normalised);
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
    caxis([lifetimeLow lifetimeHigh])
    imagewd = getframe(gcf); 
    imwrite(imagewd.cdata, [filePath,'\New Analysis',analysisType, '\Lifetime_', num2str(CurrentWavelength),'nm','\r', num2str(row),'_c', num2str(colum), '_Lifetime ', num2str(CurrentWavelength),'nm.tif']);
    
    close(gcf);

    % Plot mean lifetimes vs wavelength
    meanTauLeastSquaresVsWavelength = [];
    stdTauLeastSquaresVsWavelength = [];
    tauLeastSquaresReshaped = lifetimeDatatoAnalyse;
    
    for spectralPixel = 1:1:512
        
        meanTauLeastSquares = nanmean(abs(tauLeastSquaresReshaped(spectralPixel,:,:)),'all');
        stdDevTauLeastSquares = nanstd(abs(tauLeastSquaresReshaped(spectralPixel,:,:)),[],'all');
        meanTauLeastSquaresVsWavelength = [meanTauLeastSquaresVsWavelength meanTauLeastSquares];
        stdTauLeastSquaresVsWavelength = [stdTauLeastSquaresVsWavelength stdDevTauLeastSquares];
    end
    
    close(gcf);
    
    figure
    plot(wavelengths,meanTauLeastSquaresVsWavelength)
    hold on
    meanPlusStdVsWavelength = meanTauLeastSquaresVsWavelength + stdTauLeastSquaresVsWavelength;
    meanMinusStdVsWavelength = meanTauLeastSquaresVsWavelength - stdTauLeastSquaresVsWavelength;
    plot(wavelengths,meanPlusStdVsWavelength)
    plot(wavelengths,meanMinusStdVsWavelength)
    legend('Mean', 'Mean + Std', 'Mean - Std')
    xlabel('Wavelength (nm)')
    ylabel('tau mean')
    ylim([lifetimeLow lifetimeHigh])
    set(gcf,'color','w');
    imagewd = getframe(gcf); 
    imwrite(imagewd.cdata, [filePath,'\New Analysis',analysisType, '\Histograms\meanTau\r', num2str(row),'_c', num2str(colum),'_MeanTau.tif']);
    savefig([filePath,'\New Analysis',analysisType, '\Histograms\meanTau\r', num2str(row),'_c', num2str(colum),'_MeanTau.fig']);
    close(gcf);
    
    % plot spectrum for ROI
    close(gcf);
    figure
    spectrum = sum(sum(alphaDatatoAnalyse, 2),3);
    plot(wavelengths, spectrum)
    title('Sectrum')
    xlabel('Wavelength (nm)')
    ylabel('Counts')
    imagewd = getframe(gcf); 
    imwrite(imagewd.cdata, [filePath,'\New Analysis',analysisType, '\Histograms\meanTau\r', num2str(row),'_c', num2str(colum),'Spectrum.tif']);
    savefig([filePath,'\New Analysis',analysisType, '\Histograms\meanTau\r', num2str(row),'_c', num2str(colum),'Spectrum.fig']);
    close(gcf);
end

