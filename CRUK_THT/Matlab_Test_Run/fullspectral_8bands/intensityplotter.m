function  intensityplotter(Image, Folder)


imagesc(Image);
colormap(hot);

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
limit = [0 1];
caxis(limit);

% fig = gcf;
% truesize(fig, [256 256]);

imagewd = getframe(gcf);
imwrite(imagewd.cdata, Folder);
close(gcf);




% set(gca,'Color','k')
% imagesc(Image, 'AlphaData', bins_array_alpha_normalised);
% 
% 
% axis off
% 
% set(gcf, 'Units', 'centimeters', 'OuterPosition', [0.3, 0.3, 10, 10.6]);
% ti = [ 0 0 0 0 ];
% set(gca,'Position',[ti(1) ti(2) 1-ti(3)-ti(1) 1-ti(4)-ti(2)]);
% set(gca,'units','centimeters')
% pos = get(gca,'Position');
% set(gca,'Color','k')
% caxis([lifetimeLow lifetimeHigh])
% 
% % set(hFigure, 'MenuBar', 'none');
% % set(hFigure, 'ToolBar', 'none');
% imagewd = getframe(gcf);
% imwrite(imagewd.cdata, Folder);
% close(gcf);


end













% imagesc(Image);
% axis off
% %ti = get(gca,'TightInset')
% set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.3, 0.3, 0.3, 0.50]);
% ti = [ 0 0 0 0.09 ];
% set(gca,'Position',[ti(1) ti(2) 1-ti(3)-ti(1) 1-ti(4)-ti(2)]);
% set(gca,'units','centimeters')
% pos = get(gca,'Position');
% %ti = get(gca,'TightInset');
%
% set(gcf, 'PaperUnits','centimeters');
% set(gcf, 'PaperSize', [pos(3)+ti(1)+ti(3) pos(4)+ti(2)+ti(4)]);
% set(gcf, 'PaperPositionMode', 'manual');
% set(gcf, 'PaperPosition',[0 0 pos(3)+ti(1)+ti(3) pos(4)+ti(2)+ti(4)]);
% colormap(hot);
%
% imagewd = getframe(gcf);
% imwrite(imagewd.cdata, [Folder, '\r', num2str(row),'_c', num2str(colum),'_Intensity.tif']);
% save([Folder,'\r', num2str(row),'_c', num2str(colum), '_IntensityData.mat'],'Image')
% close(gcf);
