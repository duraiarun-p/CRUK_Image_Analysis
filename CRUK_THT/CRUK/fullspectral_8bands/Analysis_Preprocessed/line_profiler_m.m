
alllifetimeData = lifetimeImageData{1};
allAlphaData = lifetimeAlphaData{1};
imagetoplot = squeeze(allAlphaData(30,:,:));

imagesc(imagetoplot);

%
h = imfreehand;
pos = h.getPosition();
%
hold on
pointData = round(pos);
plot(pointData(:,1), pointData(:,2), 'Color', 'r', 'LineWidth', 1.5);
hold off

point = 1;
numberofPointstoLabel = 5;
labelSpaceing = size(pointData, 1) / numberofPointstoLabel;
for k = 1:labelSpaceing:size(pointData, 1)
    
text(pointData(point,1), pointData(point,2), num2str(point))
point = point+labelSpaceing;
end
%

lineData = [];

for m = 1:size(pointData, 1)

lineData(:,m) = alllifetimeData(:,pointData(m,1), pointData(m,2));

end
lineData = flip(lineData,2);
%%
figure
mesh(movmean(lineData,20,2));
set(gca, 'Color', 'K');
colormap('jet')
ylim([0 410])
caxis([1 2.8])
zlim([1 2.7])
xlim([0 size(lineData,2)])
xticks(1:labelSpaceing:size(pointData, 1));
xlabel('Point along profile');
ylabel('Wavelength');
zlabel('Lifetime');