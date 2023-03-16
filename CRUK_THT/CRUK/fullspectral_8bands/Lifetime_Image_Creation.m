function Lifetime_Image_Creation(spectral_pixel, bin, mask_threshold, sample, count_threshold, selected_data_for_subtraction, parameters_cpu, bins_array_movsum_selected_reshaped, wavelengths, lifetimeLow, lifetimeHigh, filePath, row, colum, analysisType, AlphaScalefactor, oneMinusAlpha)

    CurrentWavelength = round(spectral_pixel*0.5468 + 500);
 


%     % remove lifetime data with too few counts by threashold on peak bin
%     numberofbins = size(selected_data_for_subtraction(:,1),1);
%     selected_data_for_subtractionPeakbin = selected_data_for_subtraction(numberofbins,:);
%     binRatioforSubtraction = selected_data_for_subtractionPeakbin;
%     binRatioforSubtraction(binRatioforSubtraction<count_threshold)=0;
%     binRatioforSubtraction(binRatioforSubtraction>0)=1;
%     parameters_cpu(2,:) = parameters_cpu(2,:).*binRatioforSubtraction;
    
  
    % create mask
    array_movsum_selected = reshape(bins_array_movsum_selected_reshaped, size(bins_array_movsum_selected_reshaped, 2), size(bins_array_movsum_selected_reshaped, 1), size(bins_array_movsum_selected_reshaped, 3));
    bins_array_reshaped_one_frame = reshape(array_movsum_selected(spectral_pixel, bin, :,:),[256 256]);
    mask = (bins_array_reshaped_one_frame > mask_threshold);
    
    % Plot lifetime image
    convSize = 3;
    
%     
%     
%     tauLeastSquaresCPU = parameters_cpu(2,:)';
%     tauLeastSquaresCPU(tauLeastSquaresCPU==0)=NaN; %%% CHANGE THIS BACK TO "==0" !!!!!!
% 
% 
%     tauLeastSquaresReshaped = reshape(tauLeastSquaresCPU, [512 256 256]);

        numberofbins = size(selected_data_for_subtraction(:,1),1);
        selected_data_for_subtractionPeakbin = selected_data_for_subtraction(numberofbins,:);
        mask = selected_data_for_subtractionPeakbin;
        mask(mask<count_threshold)=0;
        mask(mask>count_threshold)=1;
        parameters_cpu(2,:) = parameters_cpu(2,:).*mask;
        tauLeastSquaresCPU = parameters_cpu(2,:); 
        tauLeastSquaresCPU(tauLeastSquaresCPU>4)=0;
        tauLeastSquaresCPU(tauLeastSquaresCPU==0)=NaN;
        tauLeastSquaresReshaped = reshape(tauLeastSquaresCPU, [512 256 256]);
    
    bins_array_alpha = selected_data_for_subtractionPeakbin;
    bins_array_alpha = reshape(bins_array_alpha, [512 256 256]);
    bins_array_alphaPixel = squeeze(bins_array_alpha(spectral_pixel, :,:));
    
    close all
    tauLeastSquaresReshapedDisplayFrame = reshape(tauLeastSquaresReshaped(spectral_pixel,:,:),[256 256]);
    image_masked = tauLeastSquaresReshapedDisplayFrame;
    %image_masked(~mask) = double(0);
    histogram(image_masked, 1000)
    %xlim([lifetimeLow lifetimeHigh+1])
    %ylim([0 1000])
    xlabel('Lifetime (ns)')
    ylabel('Counts')
    caption = sprintf('%s %s=%.2fnm',sample,'\lambda',wavelengths(spectral_pixel));
    title(caption);
    set(gcf,'color','w');
    imagewd = getframe(gcf); 
    imwrite(imagewd.cdata, [filePath,analysisType, '\Histograms\', num2str(CurrentWavelength),'nm','\r', num2str(row),'_c', num2str(colum), '_Histogram.tif']);
    savefig([filePath,analysisType,  '\Histograms\', num2str(CurrentWavelength),'nm','\r', num2str(row),'_c', num2str(colum), '_Histogram.fig']);
    save([filePath, analysisType, '\Histograms\', num2str(CurrentWavelength),'nm','\r', num2str(row),'_c', num2str(colum), '_Histogram.mat'],'image_masked')
    close(gcf);

    figure
    tauLeastSquaresReshapedDisplayFrame(tauLeastSquaresReshapedDisplayFrame>4) = 0;
    
    if oneMinusAlpha == 1
    bins_array_alpha_normalised = AlphaScalefactor*(1-bins_array_alphaPixel/max(max(bins_array_alphaPixel)));
    % try scaling / 1-alpha
    else
        bins_array_alpha_normalised = AlphaScalefactor*bins_array_alphaPixel/max(max(bins_array_alphaPixel));
    end

    imagesc(tauLeastSquaresReshapedDisplayFrame, 'AlphaData', bins_array_alpha_normalised);
    colormap(jet);
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
    imwrite(imagewd.cdata, [filePath,analysisType, '\Lifetime_', num2str(CurrentWavelength),'nm','\r', num2str(row),'_c', num2str(colum), '_Lifetime ', num2str(CurrentWavelength),'nm.tif']);
    
    close(gcf);
    figure
    imagesc(bins_array_alpha_normalised');
    title('Normalised');
    close(gcf);
    figure
    imagesc(image_masked)
    colorbar
    caxis([lifetimeLow lifetimeHigh])
    xlabel('Column')
    ylabel('Row')
    caption = sprintf('%s %s=%.2fnm',sample,'\lambda',wavelengths(spectral_pixel));
    title(caption);
    set(gcf,'color','w');
    close(gcf);
    figure
    % Compute image moving average using convolution
    B = ones(convSize,convSize)/convSize^2; 
    C = conv2(image_masked,B,'same'); 
    imagesc(C)
    colorbar
    caxis([lifetimeLow lifetimeHigh])

    % Plot mean lifetimes vs wavelength
    meanTauLeastSquaresVsWavelength = [];
    stdTauLeastSquaresVsWavelength = [];
    tauLeastSquaresReshaped = reshape(tauLeastSquaresCPU, [512 256*256]);
    
    for spectralPixel = 1:1:512
        spectralPixel;
        meanTauLeastSquares = nanmean(abs(tauLeastSquaresReshaped(spectralPixel,:,:)));
        stdDevTauLeastSquares = nanstd(abs(tauLeastSquaresReshaped(spectralPixel,:,:)));
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
    title(sample)
    ylim([0 5])
    set(gcf,'color','w');
    imagewd = getframe(gcf); 
    imwrite(imagewd.cdata, [filePath,analysisType, '\Histograms\meanTau\r', num2str(row),'_c', num2str(colum),'_MeanTau.tif']);
    savefig([filePath,analysisType, '\Histograms\meanTau\r', num2str(row),'_c', num2str(colum),'_MeanTau.fig']);
    close(gcf);
    
end

