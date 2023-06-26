close all % close all open figures otherwise they build up fast
% first convert the cells into arrays as these are easiest to work with -

intesityData = cell2mat(lifetimeAlphaData(1)); 
lifetimeData = cell2mat(lifetimeImageData(1));

%you can then plot an intensity image from anywhere along the wavelength spectrum using 
figure % make a new figure

pixelNumber = 100; % define the wavelength you want to look at
imagesc(squeeze(intesityData(pixelNumber,:,:))) 

% the "squeeze" here removes the unit dimension from the array (otherwise it ends up a 1x256x256 array which you cannot use imagesc on)
% note that ":" just means "all" so here I have selected the 100th wavelength and all of the pixels in x and y

% To plot a lifetime image that looks nice you can use the matlab transparency channel based on an intensity image, which I normally define as another variable so:

imageAlpha = squeeze(intesityData(pixelNumber,:,:))/max(max(squeeze(intesityData(pixelNumber,:,:)))); % note the division by the max, alpha goes from 0-1, anything >1 gets set to 1

%you can scale this alpha image to bring out more and eventually remove the effect of the alpha channel completely by increasing this factor (more and more of the image gets becomes Alpha of 1)

alphaScaling = 2.5;

% Then plot the image

figure % make new figure
imagesc(squeeze(lifetimeData(pixelNumber,:,:)), 'AlphaData', ImageAlpha );
colorbar % turn on the colorbar

% note that the default background colour of a matlab figure is white so when transparency is applied this is what you see. so we set this to black

set(gca, 'Color', 'k')

% the next bit selects region of interest data from the lifetime data - click 2 points on the image that cover the region of interest ( you can do more complex regions than this...)

coordianates = round(ginput(2)); % this gets the pixels for the corners of the region

% if you want to plot say the lifetime vs wavelength for a particular pixel you can select the pixel coordinates from one of the figures created above and select "data tips" from the tools menu then change pixelX and pixelY below

pixelX = 119;
pixelY  = 223;
figure % make new figure
plot(squeeze(lifetimeData(:,pixelX ,pixelY  )))

%this should plot the lifetime vs wavelength for the region of interest selected, first the lifetime data is selected from the region of interest, then the mean value of all the pixels is calculated then its plotted

regionOfInterestData = lifetimeData(:,coordianates(1):coordianates(2),coordianates(3):coordianates(4));
regionLifetimeMean = mean(mean(regionOfInterestData,2),3)

figure
plot(regionLifetimeMean);

