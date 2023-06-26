classdef hyperspectralToolBox
    methods(Static)
        
        function plot_bin_16_sorted(spectral_pixel, bin_16_array, wavelengths)
            close all
            %% Sort Bin 16 noise
            intensityBin16ForOneSpectralPixel = bin_16_array(spectral_pixel,:);
            sortedIntensityBin16 = sort(intensityBin16ForOneSpectralPixel);
            figure
            plot(sortedIntensityBin16)
            intensityCaption = sprintf('%s=%.2fnm Counts','\lambda',wavelengths(spectral_pixel));
            title(strcat(intensityCaption, ' 64 Bins summed. Sorted Bin 16'))
            xlabel('Pixel')
            ylabel('Counts')
            set(gcf,'color','w');
        end
        
        function mask = plot_bin_counts(mask_threshold, bin, spectral_pixel, array_movmean_selected, wavelengths, sample)
            
            close all
            bins_array_reshaped_one_frame = reshape(array_movmean_selected(spectral_pixel, bin, :,:),[256 256]);
            % Transpose to get correct orientation if necessary
            mask = (bins_array_reshaped_one_frame > mask_threshold);
            figure
            imagesc(mask)
            figure
            mesh(bins_array_reshaped_one_frame)
            colorbar
            xlabel('Column')
            ylabel('Row')
            caption = sprintf('%s 16 wavelength sum %s=%.2fnm',sample,'\lambda',wavelengths(spectral_pixel));
            title(caption);
            set(gcf,'color','w');
            figure
            plot(bins_array_reshaped_one_frame(150,:))
            
        end % plot_bin_counts
        
        function plot_bin16_mean(meanAllBins16VsWavelength, bin_16_array, selectedSortPosition, wavelengths)
            global improvedBin16Mean
            %close all;
            % Plot versus wavelengthSortPosition, 
            figure(1)
            plot(wavelengths, meanAllBins16VsWavelength')
            xlabel('Wavelength (nm)')
            ylabel('Counts')
            title('Mean Bin 16 vs Wavelength')
            set(gcf,'color','w');
            % Plot sorted
            figure(2)
            [sortedMeanAllBins16VsWavelength sortedMeanAllBins16VsWavelengthPositions] = sort(meanAllBins16VsWavelength);
            sortedMeanAllBins16VsWavelengthPositions(1)
            plot(sortedMeanAllBins16VsWavelength)
            xlabel('Index)')
            ylabel('Counts')
            title('Sorted Mean Bin 16')
            set(gcf,'color','w');
            figure(3)
            plot(sortedMeanAllBins16VsWavelengthPositions)
            figure(4)
            bin16FrameFromSortPosition = bin_16_array(sortedMeanAllBins16VsWavelengthPositions(selectedSortPosition),:);
            bin16FrameFromSortPositionReshaped = reshape(bin16FrameFromSortPosition, [256, 256]);
            % Compute image moving average using convolution
            B = ones(27,27)/27^2; 
            C = conv2(bin16FrameFromSortPositionReshaped,B,'same'); 
            imagesc(C)
            colorbar
            caption = sprintf('Wavelength = %f',wavelengths(sortedMeanAllBins16VsWavelengthPositions(selectedSortPosition)));
            title(caption)
            figure(5)
            histogram(bin16FrameFromSortPositionReshaped)
            figure(6)
            histogram(C)
            figure(7)
            imagesc(bin16FrameFromSortPositionReshaped)
            colorbar
            title(caption)
            figure(8)
            copyOfC = C;
            indices = find(abs(copyOfC)<0.16);
            copyOfC(indices) = 0;
            imagesc(copyOfC)
            colorbar
            title(caption)
            figure(9)
            bin16BackgroundPixels = bin_16_array(:,indices);
            bin16BackgroundPixels_mean = mean(bin16BackgroundPixels, 2);
            plot(wavelengths, bin16BackgroundPixels_mean')
            hold on
            plot(wavelengths, meanAllBins16VsWavelength')
            figure(10)
            plot(wavelengths, meanAllBins16VsWavelength' - bin16BackgroundPixels_mean')
            ylim([0 0.05])
            improvedBin16Mean = bin16BackgroundPixels_mean;
            
            
        end
        
        function plot_decays_and_bin16_mean(spectral_pixel, row, col, bins_array_3, wavelengths)
            %% Plot decays
            close all
            % Get the mean bin 16 background for all wavelengths
            allBins16 = bins_array_3(:,16,:);
            allBins16Reshaped = reshape(allBins16, [512 65536]);
            meanAllBins16VsWavelength = mean(allBins16Reshaped,2);
            maxForSpectralPixel = max(max(max(bins_array_3(spectral_pixel,:,:,:))))
            caption = sprintf('Row %.0f Column %.0f Spectral Pixel %.0f',row, col, spectral_pixel);
            decaysForSpectralPixel = bins_array_3(spectral_pixel,:,row,col);
            plot(decaysForSpectralPixel)
            title(caption);
            xlabel('Time Bin')
            ylabel('Counts')
            set(gcf,'color','w');
            grid on
            hold on
            decaysLessBackground = decaysForSpectralPixel-meanAllBins16VsWavelength(spectral_pixel);
            plot(decaysLessBackground)
            legend('Raw Counts', 'Raw Counts - Bin 16 Mean')
            disp('meanAllBins16VsWavelength')
            decaysForSpectralPixel
            decaysLessBackground
            meanAllBins16VsWavelength(spectral_pixel)
            figure
            plot(wavelengths, meanAllBins16VsWavelength')
            xlabel('Wavelength (nm)')
            ylabel('Counts')
            title('Mean Bin 16 vs Wavelength')
            set(gcf,'color','w');
            hold
            meanAllBins16VsWavelengthSmoothed64 = movmean(meanAllBins16VsWavelength',64);
            plot(wavelengths, meanAllBins16VsWavelengthSmoothed64)
            legend('Bin 16 Mean', 'Bin 16 Moving Mean (64 channels)')
            
            
        end
        
        function plot_decays_and_bin16_max(spectral_pixel, row, col, bins_array_3, wavelengths)
            %% Plot decays
            % Get the mean bin 16 background for all wavelengths
            allBins16 = bins_array_3(:,16,:);
            allBins16Reshaped = reshape(allBins16, [512 65536]);
            %meanAllBins16VsWavelength = mean(allBins16Reshaped,2);
            %minAllBins16VsWavelength = min(allBins16Reshaped,2);
            maxAllBins16VsWavelength = max(allBins16Reshaped,2);
            
            caption = sprintf('Row %.0f Column %.0f Spectral Pixel %.0f',row, col, spectral_pixel);
            decaysForSpectralPixel = bins_array_3(spectral_pixel,:,row,col);
            plot(decaysForSpectralPixel)
            title(caption);
            xlabel('Time Bin')
            ylabel('Counts')
            set(gcf,'color','w');
            grid on
            hold on
%             decaysLessBackground = decaysForSpectralPixel-meanAllBins16VsWavelength(spectral_pixel);
%             plot(decaysLessBackground)
%             legend('Raw Counts', 'Raw Counts - Bin 16 Mean')
            figure
            %plot(wavelengths, meanAllBins16VsWavelength')
            %plot(wavelengths, minAllBins16VsWavelength')
            plot(wavelengths, maxAllBins16VsWavelength')
            xlabel('Wavelength (nm)')
            ylabel('Counts')
            title('Mean Bin 16 vs Wavelength')
            set(gcf,'color','w');
            hold
            %meanAllBins16VsWavelengthSmoothed64 = movmean(meanAllBins16VsWavelength',64);
            %plot(wavelengths, meanAllBins16VsWavelengthSmoothed64)
            legend('Bin 16 Mean', 'Bin 16 Moving Mean (64 channels)', 'Bin 16 Min', 'Bin 16 Max')
            
            
        end
        
        function plot_histogram(spectral_pixel, tauLeastSquaresCPU, wavelengths)
            tauLeastSquaresReshaped = reshape(tauLeastSquaresCPU, [512 256 256]);
            close all
            tauLeastSquaresReshapedDisplayFrame = reshape(tauLeastSquaresReshaped(spectral_pixel,:,:),[256 256]);
            histogram(tauLeastSquaresReshapedDisplayFrame)
            xlim([0.75 3])
            ylim([0 2000])
            xlabel('Lifetime (ns)')
            ylabel('Counts')
            caption = sprintf('Cancer sample %s=%.2fnm','\lambda',wavelengths(spectral_pixel));
            title(caption);
            set(gcf,'color','w');
            
        end
        
        function plot_lifetime_image_histogram_and_lifetime_v_wavelength(mask, bins_array_alpha, spectral_pixel,tauLeastSquaresCPU, wavelengths, sample, convSize)
            %% Lifetime stats analysis All wavelength frames - EDIT THIS ONE!!!
            lifetimeLow = 0.8;
            lifetimeHigh = 1.2;

            tauLeastSquaresReshaped = reshape(tauLeastSquaresCPU, [512 256 256]);
            close all
            tauLeastSquaresReshapedDisplayFrame = reshape(tauLeastSquaresReshaped(spectral_pixel,:,:),[256 256]);
            image_masked = tauLeastSquaresReshapedDisplayFrame;
            image_masked(~mask) = double(0);
            histogram(image_masked, 1000)
            xlim([1 5])
            ylim([0 1000])
            xlabel('Lifetime (ns)')
            ylabel('Counts')
            caption = sprintf('%s %s=%.2fnm',sample,'\lambda',wavelengths(spectral_pixel));
            title(caption);
            set(gcf,'color','w');

            figure
            tauLeastSquaresReshapedDisplayFrame(tauLeastSquaresReshapedDisplayFrame>4) = 0;
            bins_array_alpha_normalised = 1*bins_array_alpha/max(max(bins_array_alpha));
            imagesc(tauLeastSquaresReshapedDisplayFrame', 'AlphaData', bins_array_alpha_normalised');
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
            
            
            
            %caption = sprintf('Alpha mask. %s %s=%.2fnm',sample,'\lambda',wavelengths(spectral_pixel));
            %title(caption);
            set(gcf,'color','w');
            %colorbar
            caxis([lifetimeLow lifetimeHigh])
            figure
            imagesc(bins_array_alpha_normalised');
            title('Normalised');
            figure
            imagesc(image_masked)
            colorbar
            caxis([lifetimeLow lifetimeHigh])
            xlabel('Column')
            ylabel('Row')
            caption = sprintf('%s %s=%.2fnm',sample,'\lambda',wavelengths(spectral_pixel));
            title(caption);
            set(gcf,'color','w');
            figure
            
            % Compute image moving average using convolution
            B = ones(convSize,convSize)/convSize^2; 
            %C = conv2(tauLeastSquaresReshapedDisplayFrame,B,'same'); 
            C = conv2(image_masked,B,'same'); 
            imagesc(C)
            colorbar
%             caption = sprintf('Wavelength = %f',wavelengths(sortedMeanAllBins16VsWavelengthPositions(selectedSortPosition)));
%             title(caption)
            caxis([lifetimeLow lifetimeHigh])
            % Plot mean lifetimes vs wavelength
            meanTauLeastSquaresVsWavelength = []
            stdTauLeastSquaresVsWavelength = []
            tauLeastSquaresReshaped = reshape(tauLeastSquaresCPU, [512 256*256]);
            for spectralPixel = 1:1:512
                spectralPixel;
                meanTauLeastSquares = nanmean(abs(tauLeastSquaresReshaped(spectralPixel,:,:)));
                stdDevTauLeastSquares = nanstd(abs(tauLeastSquaresReshaped(spectralPixel,:,:)));
                meanTauLeastSquaresVsWavelength = [meanTauLeastSquaresVsWavelength meanTauLeastSquares];
                stdTauLeastSquaresVsWavelength = [stdTauLeastSquaresVsWavelength stdDevTauLeastSquares];
            end
            
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
            %title('Margin sample')
            %title('Normal sample')
            title(sample)
            ylim([0 5])
            set(gcf,'color','w');
            
        end
        
        function plot_lifetime_images_with_row_and_column_charts(spectral_pixel, tauLeastSquaresCPU, wavelengths)
            %% Images and figures for lifetime least squares
            %numberOfSpectralFrames = 1;
            numberOfSpectralFrames = 512;
            lifetimeData = reshape(tauLeastSquaresCPU, [numberOfSpectralFrames 256 256]);
            % Frame for single spectral pixel
            %lifetimeData = reshape(tauLeastSquares, [1 256 256]);
            figure
            % Single frame for test
            caption = sprintf('Least squares 64 summed bins %s=%.2fnm Lifetime','\lambda',wavelengths(spectral_pixel));
            % Multiple frames
            % lifetimeSpectraFrame = reshape(lifetimeDataFlippedRows(spectral_pixel,:,:),[256 256]);
            % Single frame
            lifetimeSpectraFrame = reshape(lifetimeData(spectral_pixel,:,:),[256 256]);
            im = mat2gray(lifetimeSpectraFrame, [2 4]);
            imshow(im)
            figure
            BW = edge(im, 'Canny', 0.35);
            imshow(BW)
            figure
            histogram(lifetimeSpectraFrame,1000);
            xlim([1 3.5])
            xlabel('Lifetime (ns)')
            ylabel('Counts')
            title(caption);
            set(gcf,'color','w');
            figure
            imagesc(lifetimeSpectraFrame)
            xlabel('Column Confocal Pixel')
            ylabel('Row Confocal Pixel')
            title(caption);
            colorbar
            caxis([0.0 3.5]);
            set(gcf,'color','w');
            figure
            mesh(lifetimeSpectraFrame)
            %colormap copper
            colorbar
            caxis([0 3.2]);
            zlim([0 3.2]);
            xlabel('Column Confocal Pixel')
            ylabel('Row Confocal Pixel')
            zlabel('Lifetime (ns)')
            set(gcf,'color','w');
            figure
            smooth = 20
            plot(movmean(lifetimeSpectraFrame(190,:),smooth))
            hold
            plot(movmean(lifetimeSpectraFrame(200,:),smooth))
            plot(movmean(lifetimeSpectraFrame(210,:),smooth))
            plot(movmean(lifetimeSpectraFrame(220,:),smooth))
            legend('Row 190', 'Row 200', 'Row 210', 'Row 220')
            xlabel('Column Confocal Pixel')
            ylabel('Lifetime (ns)')
            title(caption);
            set(gcf,'color','w');
            figure
            plot(movmean(lifetimeSpectraFrame(:,70),smooth))
            hold
            plot(movmean(lifetimeSpectraFrame(:,80),smooth))
            plot(movmean(lifetimeSpectraFrame(:,90),smooth))
            plot(movmean(lifetimeSpectraFrame(:,100),smooth))
            xlabel('Row Confocal Pixel')
            ylabel('Lifetime (ns)')
            legend('Col 70', 'Col 80', 'Col 90', 'Col 100')
            title(caption);
            set(gcf,'color','w');
        end
        
        function plot_and_save_lifetime_images_and_histograms(tauLeastSquaresCPU, wavelengths)
            %% Lifetime least squares - loop to generate lifetime movie and all wavelengths histogram
            lifetimeData = reshape(tauLeastSquaresCPU, [512 256 256]);
            figure('Position', [10 10 1500 500])
            for spectral_pixel = 1:1:512
                spectral_pixel
                caption = sprintf('Least squares 64 summed bins %s=%.2fnm Lifetime','\lambda',wavelengths(spectral_pixel));
                lifetimeSpectraFrame = reshape(lifetimeData(spectral_pixel,:,:),[256 256]);
                %figure
                % Lifetime frame for this wavelength
                %figure
                subplot(1, 2, 1);
                imagesc(lifetimeSpectraFrame)
                xlabel('Column Confocal Pixel')
                ylabel('Row Confocal Pixel')
                title(caption);
                colorbar
                caxis([2.0 3.0]);
                set(gcf,'color','w');
                %filename = sprintf('%s_%d.png','E:\Inverted_Kronoscan\20190522\HistMode_no_pixel_binning\cr74_actually74_cancer_full_90%\lifetime_images\image',spectral_pixel);
                %saveas(gcf,filename)
                % Histogram for this wavelength
                subplot(1, 2, 2);
                histogram(lifetimeSpectraFrame,1000);
                xlim([0.5 4.0])
                %ylim([0 500])
                xlabel('Lifetime (ns)')
                ylabel('Counts')
                title(caption);
                set(gcf,'color','w');
                marginPath = 'E:\Inverted_Kronoscan\20190522\HistMode_no_pixel_binning\cr72_cancer_full_90%\lifetime_images_high_quality\image';
                cancerPath = 'E:\Inverted_Kronoscan\20190522\HistMode_no_pixel_binning\cr74_actually74_cancer_full_90%\lifetime_images_high_quality\image';
                normalPath = 'E:\Inverted_Kronoscan\20190522\cr74_normal_full_90%\lifetime_images_high_quality\image';
                convollariaPath = 'E:\Inverted_Kronoscan\20190516\convalaria20xhist3\lifetime_images_high_quality\image';
                filename = sprintf('%s_%d.png',normalPath,spectral_pixel);
                saveas(gcf,filename)
                
            end
        end
        
        function plot_and_save_Raman_images(single_bin_array_reshaped_less_background, wavelengths)
            %% Raman - loop to generate Raman movie 
            
            figure('Position', [10 10 700 500])
            useCustomMap = 1;
            for spectral_pixel = 1:1:512
                spectral_pixel
                caption = sprintf('Normal Sample Bin 15 %s=%.2fnm Raman','\lambda',wavelengths(spectral_pixel));
                ramanFrame = reshape(single_bin_array_reshaped_less_background(spectral_pixel,:,:),[256 256]);
                %figure
                % Raman frame for this wavelength
                %figure
                imagesc(ramanFrame)
                %view(180,90)
                xlabel('Column')
                ylabel('Row')
                title(caption);
                if useCustomMap == 1
                    custom_map =[]
                    levels = 256;
                    for i = 1:1:levels
                        custom_map = [custom_map [i*spectrumRGB(wavelengths(spectral_pixel))/levels]];
                    end
                    custom_map = transpose(reshape(custom_map, [3 levels]));
                    colormap(gca, custom_map);
                end
                colorbar
                caxis([0 35]);
                set(gcf,'color','w');
                shading interp;
                pause(0.001);

                ramanNormalPath = 'E:\Inverted_Kronoscan\20190522\Raman_Normal\lifetime_images_high_quality\image';
                filename = sprintf('%s_%d.png',ramanNormalPath,spectral_pixel);
                saveas(gcf,filename)
                
            end
        end
        
        function plot_pixels(spectral_pixel, tauLeastSquaresCPU, wavelengths)
            tauLeastSquaresReshaped = reshape(tauLeastSquaresCPU, [512 256 256]);
            %% Line graphs - lifetime
            % Plot 4 or so pixels vs wavelength
            figure
            col = 64;
            row = 64;
            lifetimeDataRowColumn1 = tauLeastSquaresReshaped(:,row,col)
            col = 128;
            row = 128;
            lifetimeDataRowColumn2 = tauLeastSquaresReshaped(:,row,col)
            col = 192;
            row = 192;
            lifetimeDataRowColumn3 = tauLeastSquaresReshaped(:,row,col)
            col = 256;
            row = 256;
            lifetimeDataRowColumn4 = tauLeastSquaresReshaped(:,row,col)
            plot(wavelengths,lifetimeDataRowColumn1)
            hold
            plot(wavelengths,lifetimeDataRowColumn2)
            plot(wavelengths,lifetimeDataRowColumn3)
            plot(wavelengths,lifetimeDataRowColumn4)
            set(gcf,'color','w');
            xlim([520 740])
            xlabel('Wavelength (nm)')
            ylim([0 2.5])
            ylabel('Lifetime (ns)')
            legend('(64,64)','(128,128)', '(192,192)', '(256,256)')
            title('4 Pixels')
            % Plot lifetime across margin row
            figure
            row = 25;
            lifetimeDataRow1 = reshape(tauLeastSquaresReshaped(spectral_pixel,row,:),[1 256]);
            row = 128;
            lifetimeDataRow2 = reshape(tauLeastSquaresReshaped(spectral_pixel,row,:),[1 256]);
            row = 192;
            lifetimeDataRow3 = reshape(tauLeastSquaresReshaped(spectral_pixel,row,:),[1 256]);
            plot(movmean(lifetimeDataRow1, 10))
            hold
            plot(movmean(lifetimeDataRow2, 10))
            plot(movmean(lifetimeDataRow3, 10))
            xlabel('Pixel')
            ylim([0 2.5])
            ylabel('Lifetime (ns)')
            set(gcf,'color','w');
            legend('Row 25','Row 128', 'Row 192')
            title('Row Lifetimes')
            
        end
        
        function plot_row_and_column_vs_wavelength(tauLeastSquaresCPU, row, col, wavelengths)
            %% Spectral variation by row, column and pixel
            %figure('Position', [10 10 300 200])
            close all
            colCaption = sprintf('Column %.0f',col);
            rowCaption = sprintf('Row %.0f',row);
            tauLeastSquaresCPUFrames = reshape(tauLeastSquaresCPU,[512 256 256]);
            lifetimeAllSpectraPixelColumn = tauLeastSquaresCPUFrames(:,:,col);
            imagesc(lifetimeAllSpectraPixelColumn')
            title(colCaption)
            colorbar
            caxis([0 2.4]);
            xlabel('Wavelength (nm)')
            ylabel('Row Pixel')
            atick = 1:100:512; %assuming you want the ticks in the centre of each block
            set(gca,'XTick',atick);
            xTickLabels = wavelengths(1):50:wavelengths(512);
            xTickLabels = xTickLabels - 0.729;
            set(gca,'XTickLabel', xTickLabels);
            set(gcf,'color','w');
            % Row image
            figure
            %lifetimeAllSpectraPixelRow = reshape(lifetimeDataFlippedRows(:,row,:),[512 256]);
            lifetimeAllSpectraPixelRow = tauLeastSquaresCPUFrames(:,row,:);
            lifetimeAllSpectraPixelRowReshaped = reshape(lifetimeAllSpectraPixelRow, [512 256]);
            imagesc(lifetimeAllSpectraPixelRowReshaped')
            xlabel('Wavelength (nm)')
            ylabel('Column Pixel')
            title(rowCaption)
            colorbar
            caxis([0 2.4]);
            atick = 1:100:512; %assuming you want the ticks in the centre of each block
            set(gca,'XTick',atick);
            xTickLabels = wavelengths(1):50:wavelengths(512);
            xTickLabels = xTickLabels - 0.729;
            set(gca,'XTickLabel', xTickLabels);
            set(gcf,'color','w');
            
        end
        
        function plot_single_pixel_decay(bins_array, row, col, bin_width, spectral_pixel, wavelengths)
            selected_bins = 1:16;
            bin_width = 0.8; % In ns
            timeBins = selected_bins*bin_width;
            bins_for_pixel = bins_array(spectral_pixel,:,row,col);
            figure
            plot(timeBins, bins_for_pixel);
            set(gcf,'color','w');
            xlabel('Time (ns)')
            ylabel('Counts')
            caption = sprintf('Row = %.0f Column = %.0f %s=%.2fnm ',row, col, '\lambda',wavelengths(spectral_pixel));
            title(caption)
        end
        
    end
end