compressedVideo = VideoWriter('D:\Convalaria\testvid.avi'); 
    %set the final video frame rate
    compressedVideo.FrameRate = 20;
    compressedVideo.Quality = 50;

    % open the video file ready to start adding frames
    open(compressedVideo);

for l = 1:410
    
    wavelength = 500 +0.549*l;
    
imageDatatotest = squeeze(ImageData(l,:,:));
alphachannel = 3*squeeze(AlphaDataAll(l,:,:))/max(max(squeeze(AlphaDataAll(l,:,:))));
imageDatatotest = interp2(imageDatatotest,2);
alphachannel = interp2(alphachannel,2);

for i = 1:2:256
    n=i;
    
iamgeDataShifted = imageDatatotest(:,n);
imageDatatotest(:,n) = circshift(iamgeDataShifted,-2);
alphachannelShifted = alphachannel(:,n);
alphachannel(:,n) = circshift(alphachannelShifted,-2);
end

 imageDatatotest = interp2(imageDatatotest,-2);
 alphachannel = interp2(alphachannel,-2);

%imagesc(imageDatatotest)

imagesc(imageDatatotest', 'AlphaData', alphachannel')
caxis([0.2 2.5])
colormap(modifiedJet)
colorbar
set(gca,'Color','K')
title([num2str(wavelength), 'nm'])
pause(0.1)



 F = getframe(gcf);
    [X, Map] = frame2im(F);
    writeVideo(compressedVideo, X)
end

close(compressedVideo);