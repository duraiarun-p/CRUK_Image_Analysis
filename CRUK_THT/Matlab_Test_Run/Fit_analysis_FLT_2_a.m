
tiles=length(allIntensityImages);
plotr=sqrt(tiles);
% for spect=1:20
% for tile=1:tiles
% % I_4D=allIntensityImages{1,tile};
% % I=squeeze(I_4D);
% % h=figure(10);
% % subplot(plotr,plotr,tile)
% % imshow(I,[]);
% 
% I1_4D=lifetimeImageData{1,tile};
% % I1=squeeze(sum(I1_4D,1));
% I1=squeeze(I1_4D(4,:,:));%t=1
% % h1=figure(21);
% subplot(plotr,plotr,tile)
% imshow(I1,[]);
% 
% % I1_4D=lifetimeImageData{1,tile};
% % I1=squeeze(sum(I1_4D,1));
% I1=squeeze(I1_4D(6,:,:));%t=1
% % h1=figure(22);
% subplot(plotr,plotr,tile)
% imshow(I1,[]);
% 
% % I1_4D=lifetimeImageData{1,tile};
% % I1=squeeze(sum(I1_4D,1));
% I1=squeeze(I1_4D(4,:,:));%t=1
% % h1=figure(23);
% subplot(plotr,plotr,tile)
% imshow(I1,[]);
% end
% end



    
for spect=1:length(lambdas)
    figure(spect+50);

% I1=squeeze(sum(I1_4D,1));

    for tile=1:tiles
        I1_4D=lifetimeImageData{1,tile};
        I1=squeeze(I1_4D(spect,:,:));
subplot(plotr,plotr,tile)
imshow(I1,[]);
    end
end