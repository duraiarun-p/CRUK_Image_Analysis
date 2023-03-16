function  [compressedVideo] =  lifetimeplotter_video(Image, bins_array_alpha_normalised, lifetimeLow, lifetimeHigh, compressedVideo,CurrentWavelength, cmap)


    hFigure = figure;
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.3, 0.3, 0.8, 0.60])
    subplot(2,2,[1,3])
    imagesc(Image, 'AlphaData', bins_array_alpha_normalised);
    %imagesc(Image);
    set(subplot(2,2,[1,3]),'Color','K')

    caxis([lifetimeLow lifetimeHigh])
    colormap('jet');
    colorbar
    subplot(2,2,[2,4])
    
    [histogramValues, edges] = histcounts(Image);
    edgeNumber=size(edges,2)-1;
    area(edges(1:edgeNumber), histogramValues,'EdgeColor', 'none');
    xlim([lifetimeLow lifetimeHigh])
    xlabel('Lifetime (ns)')
    ylabel('Counts')
    title([num2str(CurrentWavelength), 'nm']);

    F = getframe(gcf);
    [X, Map] = frame2im(F);
    writeVideo(compressedVideo, X)


    close(gcf);


end






