%% Plot global normalized Alpha Lifetime plots, - can be run independently 
% if "LifetimeAlphaData"  and "LifetimeImageData" are loaded  along 
% with runing the  initialiation section of the script

% setup a video file to populate
    filename = fullfile(newFolderLifetimeData, '\processed_video3avi_noaphlpa');
    compressedVideo = VideoWriter(filename); %this one is compressed! = ~100kB/Frame
%set the final video frame rate
    compressedVideo.FrameRate = 10;
    compressedVideo.Quality = 30;
% open the video file ready to start adding frames
    open(compressedVideo);

NormalisedAlpha=[];
for z = 1:Wavelength
    row = 0;
    for l = 1:numberofRows
        row = row + 1;
        colum = 0;
        
        for k = 1:numberofColums
            colum = colum + 1;
            imageNumber = row+colum-1 + ((row-1)*(numberofColums-1));
            AlphaDataWavelength =LifetimeAlphaData{z,imageNumber};
            normalisationValue= max(max(AlphaDataWavelength));
            NormalisedAlpha(z,imageNumber) = normalisationValue;
        end
    end
end

    a=0;
for z = 10:pixeldivider:410
    row = 0;
    a=a+1;
    CurrentWavelength = round((z*0.5468 + 500),2);
    OverallNormalisedAlphaFactor = max(NormalisedAlpha(a,:));
    for l = 1:numberofRows
        row = row + 1;
        colum = 0;
        
        for k = 1:numberofColums
            colum = colum + 1;
            imageNumber = row+colum-1 + ((row-1)*(numberofColums-1));
            ImagetoPlot =LifetimeImageData{a,imageNumber};
            
            if oneMinusAlpha == 1
            AlphatoPlot = AlphaScalefactor*(1-(LifetimeAlphaData{a,imageNumber}/OverallNormalisedAlphaFactor));
            folder = [filePath,analysisType, '\LifetimeNormalised_', num2str(CurrentWavelength),'nm','\r', num2str(row),'_c', num2str(colum), '_Lifetime ', num2str(CurrentWavelength),'nm.tif'];
            lifetimeplotter_video(ImagetoPlot, folder, AlphatoPlot, lifetimeLow, lifetimeHigh)
            else
            AlphatoPlot = AlphaScalefactor*LifetimeAlphaData{a,imageNumber}/OverallNormalisedAlphaFactor;
            folder = [filePath,analysisType, '\LifetimeNormalised_', num2str(CurrentWavelength),'nm','\r', num2str(row),'_c', num2str(colum), '_Lifetime ', num2str(CurrentWavelength),'nm.tif'];
            [compressedVideo] =  lifetimeplotter_video(ImagetoPlot, folder, AlphatoPlot, lifetimeLow, lifetimeHigh, compressedVideo, CurrentWavelength);
            end
        end
    end
end
close(compressedVideo);
