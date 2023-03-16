function  [compressedVideo] =  lifetimeplotter_video(Image, bins_array_alpha_normalised, lifetimeLow, lifetimeHigh, compressedVideo,CurrentWavelength, bins_array_movsum_selected_reshaped, bin, mask_threshold, spectral_pixel, cmap)


    % create mask
%     array_movsum_selected = reshape(bins_array_movsum_selected_reshaped, size(bins_array_movsum_selected_reshaped, 2), size(bins_array_movsum_selected_reshaped, 1), size(bins_array_movsum_selected_reshaped, 3));
%     bins_array_reshaped_one_frame = reshape(array_movsum_selected(spectral_pixel, bin, :,:),[256 256]);
%     mask = bins_array_reshaped_one_frame;
%     mask(mask<mask_threshold)=0;
%     mask(mask>mask_threshold)=1;
%     image_masked = Image*mask;
    
      

    hFigure = figure;
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.3, 0.3, 0.8, 0.60])
    subplot(2,2,[1,3])
    
    B = ones(9,9)/9^2; 
    C = conv2(Image,B,'same'); 
    
    imagesc(C, 'AlphaData', bins_array_alpha_normalised);
    %imagesc(Image);
    set(subplot(2,2,[1,3]),'Color','K')

    caxis([lifetimeLow lifetimeHigh])
    colormap(cmap);
    colorbar
    shading interp
    subplot(2,2,[2,4])
    
    [histogramValues, edges] = histcounts(Image);
    edgeNumber=size(edges,2)-1;
    area(edges(1:edgeNumber), histogramValues,'EdgeColor', 'none');
    xlim([lifetimeLow lifetimeHigh])
    xlabel('Lifetime (ns)')
    ylabel('Counts')
    title([num2str(CurrentWavelength), 'nm']);
    
%     imagewd = getframe(gcf); 
%     imwrite(imagewd.cdata, folder);
    F = getframe(gcf);
    [X, Map] = frame2im(F);
    writeVideo(compressedVideo, X)


    close(gcf);


end






