function  plotter(Image, Folder, row, colum, climit)

hFigure = figure;
imagesc(Image);
axis off

set(gcf, 'Units', 'centimeters', 'OuterPosition', [0.3, 0.3, 10, 10.6]);
ti = [ 0 0 0 0 ];
set(gca,'Position',[ti(1) ti(2) 1-ti(3)-ti(1) 1-ti(4)-ti(2)]);
set(gca,'units','centimeters')
pos = get(gca,'Position');

colormap(hot);
caxis(climit);
set(hFigure, 'MenuBar', 'none');
set(hFigure, 'ToolBar', 'none');

%set(gcf,'PaperUnits','centimeters','PaperPosition',[0 0 10 10])
imagewd = getframe(hFigure);
imwrite(imagewd.cdata, [Folder, '\r', num2str(row),'_c', num2str(colum),'_Intensity.tif']);
save([Folder,'\r', num2str(row),'_c', num2str(colum), '_IntensityData.mat'],'Image')
close(hFigure);



end

