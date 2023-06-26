hFigure = figure;
imagesc(AllIntensityImagesNormalised{5});
axis off

set(gcf, 'Units', 'centimeters', 'OuterPosition', [0.3, 0.3, 10, 10.6]);
ti = [ 0 0 0 0 ];
set(gca,'Position',[ti(1) ti(2) 1-ti(3)-ti(1) 1-ti(4)-ti(2)]);
set(gca,'units','centimeters')
pos = get(gca,'Position');
%set(gcf, 'PaperUnits','centimeters');
% set(gcf, 'PaperSize', [pos(3)+ti(1)+ti(3) pos(4)+ti(2)+ti(4)]);
% set(gcf, 'PaperPositionMode', 'manual');
% set(gcf, 'PaperPosition',[0 0 pos(3)+ti(1)+ti(3) pos(4)+ti(2)+ti(4)]);
colormap(hot);

set(hFigure, 'MenuBar', 'none');
set(hFigure, 'ToolBar', 'none');

%set(gcf,'PaperUnits','centimeters','PaperPosition',[0 0 10 10])
imagewd = getframe(hFigure);
fileName = strcat(newFolderIntesnityNormalised,'\TestIntensityData.jpg');
imwrite(imagewd.cdata, fileName);
close(hFigure)