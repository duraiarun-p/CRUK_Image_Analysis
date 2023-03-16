%%test analysis script
%% SETUP PERAMITERS
% number of folders by name, before data starts, remember "." and ".."
% this is for folders within the "HistMode_no_pixel_binning" folder
% the folder selected in the popup should be one level above this folder.
numberofNondataFolders = 2;

%setnumberofrows/colums - to match data as recorded, this will effect the
%number of files read in and the naming of output files
numberofRows = 1;
numberofColums = 1;

frame_size_x = 256;

% if reading in a single image and you want to perform a region of
% intersest analysis set to 1 - STILL TO BE IMPLEMENTED
regionOfInterest = 1;

%Set name for output toplevel folder where data is saved

analysisType = '\All_section3'; 
                                                            
AlphaMask = 1; % set to 0 to plot with no Aplha masking applied - TO BE IMPLEMENTED
AlphaScalefactor = 1; % Scaling factor for alpha mask
oneMinusAlpha = 0; %set to 1 for a 1-Alpha plot

%scaling for Intesnity normalisation, increase if needed
scalingFactorIntensity = 1.2;

%set the wavelength range to look over (need to convet drom wavelength 1 =
%500 nm, 512 = 780 nm)
firstSensorPixel = 1;
lastSensorPixel = 410;
numberOfwavelengthstoPlot = 6; % number of wavelengths to fit beteen 500 and 720nm, evenly spaced

% if you want to only want to create the lifetime data cubes:
% numberOfwavelengthstoPlot = 512,firstSensorPixel = 1 ,lastSensorPixel =
% 512 then set all the ploting / video options below to 0

plotImages = 1; % set to 1 to plot and save lifetime images set to 0 to simply save data
plotNormalisedImages = 0; % set to 1 to plot and save normalised lifetime images set to 0 to simply save data
createVideo = 1; % set to 1 to create a video of the computed lifetime images with histograms
videoQuality = 60; % set between 1 and 100 if output video too large
frameRate=50; % 45-60 works well for full spectral


%peramiters for plotting=
bin = 2; % for alpha mask
sample = 'test';
lifetimeLow = 0.5; % for stained, 0.7,  1.5 for unstained, MHorrick 1
lifetimeHigh = 2.8; % for stained, 1.7,  2.8 for unstained, MHorrricks 2
      

% Load file path and find number of folders - 1 level deap to workspaces!!
currentFolder = pwd;
filePath = uigetdir;

newFolderIntesnity = [filePath, '\New Analysis', analysisType, '\Intensity'];
mkdir(newFolderIntesnity);
newFolderIntesnityNormalised = [filePath,'\New Analysis',  analysisType, '\Intensity\Normalised'];
mkdir(newFolderIntesnityNormalised);
newFolderHistograms = [filePath,'\New Analysis',  analysisType, '\Histograms'];
mkdir(newFolderHistograms);
newFolderMeanTau = [filePath,'\New Analysis',  analysisType, '\Histograms\meanTau'];
mkdir(newFolderMeanTau);
newFolderLifetimeData = [filePath, '\New Analysis', analysisType, '\Lifetime_Data'];
mkdir(newFolderLifetimeData);

pixeldivider = round((lastSensorPixel-firstSensorPixel)/numberOfwavelengthstoPlot);
lastSensorPixel = pixeldivider*numberOfwavelengthstoPlot;


Wavelength = 0;
for i = firstSensorPixel:pixeldivider:lastSensorPixel
    Wavelength = Wavelength +1;
    Wave  = round(i*0.5468 + 500);
    
    if plotImages == 1
    newFolderLifetimeImages = [filePath,'\New Analysis',  analysisType,  '\Lifetime_', num2str(Wave), 'nm'];
    mkdir(newFolderLifetimeImages);
    
    newFolderHistogramsData = [filePath,'\New Analysis',  analysisType, '\Histograms\', num2str(Wave),'nm'];
    mkdir(newFolderHistogramsData);
    end
    
    if plotNormalisedImages == 1
    newFolderLifetimeImages = [filePath,'\New Analysis',  analysisType,  '\LifetimeNormalised_', num2str(Wave), 'nm'];
    mkdir(newFolderLifetimeImages);
    end 


end

pause(0.1)

%create Meta Data File

firstWavelength = 0.549*firstSensorPixel + 500;
lastWavelength = 0.549*lastSensorPixel + 500;
AnalysedfolderName = split(filePath,"\");
metaDataFolderName = strcat('\metaData_', string(AnalysedfolderName(2)), '.csv');
metaData = {};
metaData{1} = ["Folder Analysed " ,  AnalysedfolderName(2)];
metaData{2} = ["Alpha Mask Enabled " , num2str(AlphaMask)];
metaData{3} = ["Alpha Scale Factor " , num2str(AlphaScalefactor)];
metaData{4} = ["1-Alpha Enabled " , num2str(oneMinusAlpha)];
metaData{5} = ["Number of Wavelengths Analysed " , num2str(numberOfwavelengthstoPlot)];
metaData{6} = ["Starting Wavelength " , num2str(firstWavelength)];
metaData{7} = ["Last Wavelength " , num2str(lastWavelength)];
metaData{8} = ["Short Lifetime for Plots " , num2str(lifetimeLow)];
metaData{9} = ["Long Lifetime for Plots " , num2str(lifetimeHigh)];
metaData{10} = ["Vidio Frame Rate " , num2str(frameRate)];
metaData{11} = ["Vidio Compression (%) " , num2str(videoQuality)];
metaData = splitvars(cell2table(metaData'));
metaData.Properties.VariableNames = {'Variable' 'Value'};
writetable(metaData,strcat(filePath,'\New Analysis',  analysisType , metaDataFolderName));

%%











%%
%loop through images in date order, assumes the data was recorded in row by
%row with the same starting point, data saved to the corresponding folder,
%data workspaces must be only 1 level deap from the main folder
allIntensityImages={numberofRows, numberofColums};
lifetimeDatatoAnalyse={numberofRows, numberofColums};
alphaDatatoAnalyse={numberofRows , numberofColums};
row = 0;

%move to and load worspace from 1st subfolder
currDir = [filePath,'\Lifetime_data'];
cd(currDir)
disp('Loading Processed Data:')
load('LifetimeImageData.mat')
load('LifetimeAlphaData.mat')
%return to matlab scripts directory
cd(currentFolder)

for r = 1:numberofRows
    row = row + 1;

    colum = 0;
    for k = 1:numberofColums
        colum = colum + 1;
        disp('row')
        disp(row)
        disp('column')
        disp(colum)
       
        fileNumber = row+colum-1 + ((row-1)*(numberofColums-1))+numberofNondataFolders;
        imageNumber = row+colum-1 + ((row-1)*(numberofColums-1));
        
        disp(strcat('Processing Image' , num2str(imageNumber)));
        imageLifetimeData = lifetimeImageData{row,colum};
        imageAlphaData = lifetimeAlphaData{row,colum};

       % If region of interest selected
        
       if regionOfInterest == 1

            [intensity_image] = Intensity_Image_Summation(imageAlphaData);

            hFigure = figure;


%             imagesc(intensity_image);
%             caxis([0 max(max(intensity_image))/scalingFactorIntensity])

            imagesc(squeeze(imageAlphaData(30,:,:))');


            set(gcf, 'Units', 'centimeters', 'OuterPosition', [15, 5, 20, 20.6]);
            ti = [ 0 0 0 0 ];
            set(gca,'Position',[ti(1) ti(2) 1-ti(3)-ti(1) 1-ti(4)-ti(2)]);
            set(gca,'units','centimeters')
            pos = get(gca,'Position');
            colormap('jet');
            set(hFigure, 'MenuBar', 'none');

            % Use input of 2 points to create retangular region of interest    
            [X, Y]= ginput(2);
            xmin=round(min(X));
            xmax=round(max(X));
            ymin=round(min(Y));
            ymax=round(max(Y));
            pause(0.1);
            close(hFigure);
            lifetimeDatatoAnalyse{row,colum} = imageLifetimeData(:,xmin:xmax,ymin:ymax);
            alphaDatatoAnalyse{row,colum} = imageAlphaData(:,xmin:xmax,ymin:ymax);
            pixelnumber=(xmax-xmin+1)*(ymax-ymin+1);

        else
            lifetimeDatatoAnalyse{row,colum} = imageLifetimeData;
            alphaDatatoAnalyse{row,colum} = imageAlphaData;
        end
        
        
        disp('Producing Intensity Image')

        % Produce and save intensity images
        [intensity_image] = Intensity_Image_Summation(alphaDatatoAnalyse{row,colum});
        climit = 'auto';
        plotter(intensity_image, newFolderIntesnity, row, colum, climit)
        
        allIntensityImages{row, colum} = intensity_image;
               
        %Calculate wavelength axis
        [wavelengths,wavenumbers] = Wavelength_Calculator();
        
        if plotImages == 1
            disp('Producing Lifetime Images and Plots for wavelength:')
        else
            disp('Producing Datacubes')
        end
        % Produce lifetime plots and histograms for various wavelengths
        

        wavelengthnumber = 0;
        for i = firstSensorPixel:pixeldivider:lastSensorPixel
            
            wavelengthnumber = wavelengthnumber+1;
            spectral_pixel = i;
            
            if plotImages == 1
            Currentwavelength = i*0.549 + 500;
            disp(Currentwavelength)
            Lifetime_Image_Creation(spectral_pixel, lifetimeDatatoAnalyse{row,colum}, alphaDatatoAnalyse{row,colum},  wavelengths, lifetimeLow, lifetimeHigh, filePath, row, colum,analysisType, AlphaScalefactor, oneMinusAlpha);
            end
                
            if plotImages == 0
            tauLeastSquaresReshapedDisplayFrame = reshape(tauLeastSquaresReshaped(spectral_pixel,:,:),[256 256]);
            array_movsum_selected = reshape(bins_array_movsum_selected_reshaped, size(bins_array_movsum_selected_reshaped, 2), size(bins_array_movsum_selected_reshaped, 1), size(bins_array_movsum_selected_reshaped, 3));
            bins_array_alpha = reshape(array_movsum_selected(spectral_pixel, bin, :),[256 256]);
            end

        end


    end
           


end

%% plot normalized intensity image - can be run independently if "allIntensityImages"
% is loaded and and the parameter initialisation section of the script is
% run
if plotNormalisedImages == 1
AllnormalisationValue =[];

row = 0; 
for l = 1:numberofRows
    row = row + 1;
    colum = 0;
    for k = 1:numberofColums
        colum = colum + 1;
        imageNumber = row+colum-1 + ((row-1)*(numberofColums-1));
        ImagetoData =allIntensityImages{row,colum};
        normalisationValue= max(max(ImagetoData));
        AllnormalisationValue(imageNumber) = normalisationValue;
    end
end
overallNormalisationValue = max(AllnormalisationValue);
overallNormalisationValue = overallNormalisationValue/scalingFactorIntensity; 
row = 0; 
for l = 1:numberofRows
    row = row + 1;
    colum = 0;
    for k = 1:numberofColums
        colum = colum + 1;
        imageNumber = row+colum-1 + ((row-1)*(numberofColums-1));
        ImagetoPlot =allIntensityImages{row,colum};
        IntensityImagesNormalised=ImagetoPlot/overallNormalisationValue;
        AllIntensityImagesNormalised{row,colum}= IntensityImagesNormalised;
        climit = [0 1];
        plotter(IntensityImagesNormalised, newFolderIntesnityNormalised, row, colum, climit);
    end
end

save([newFolderIntesnityNormalised,'\AllIntensityImagesNormalised.mat'],'AllIntensityImagesNormalised')
else
end
%%
save([newFolderIntesnityNormalised,'\AllIntensityData.mat'],'allIntensityImages')
%% Plot global normalized Alpha Lifetime plots, - can be run independently 
% if "LifetimeAlphaData"  and "LifetimeImageData" are loaded  along 
% with runing the  initialiation section of the script

if plotNormalisedImages == 1
NormalisedAlphaData=[];
for z = 1:Wavelength
    row = 0;
    for l = 1:numberofRows
        row = row + 1;
        colum = 0;
        
        for k = 1:numberofColums
            colum = colum + 1;
            imageNumber = row+colum-1 + ((row-1)*(numberofColums-1));
            imageAlphaData = lifetimeAlphaData{row,colum};
            AlphaDataWavelength =imageAlphaData(z,:,:);
            normalisationValue= max(max(AlphaDataWavelength));
            NormalisedAlphaData(z,imageNumber) = normalisationValue;
        end
    end
end

    a=0;
for z = firstSensorPixel:pixeldivider:lastSensorPixel
    row = 0;
    a=a+1;
    CurrentWavelength = round(z*0.5468 + 500);
    OverallNormalisedAlphaFactor = max(NormalisedAlphaData(a,:));
    for l = 1:numberofRows
        row = row + 1;
        colum = 0;
        
        for k = 1:numberofColums
            colum = colum + 1;
            imageNumber = row+colum-1 + ((row-1)*(numberofColums-1));
            ImageData =lifetimeImageData{imageNumber};
            AlphaData = lifetimeAlphaData{imageNumber};
            ImagetoPlot = squeeze(ImageData(a, : , :));
            
            if oneMinusAlpha == 1
            AlphaData = squeeze(AlphaData(a,:,:));
            AlphatoPlot = AlphaScalefactor*(1-(AlphaData/OverallNormalisedAlphaFactor));
            folder = [filePath,analysisType, '\LifetimeNormalised_', num2str(CurrentWavelength),'nm','\r', num2str(row),'_c', num2str(colum), '_Lifetime ', num2str(CurrentWavelength),'nm.tif'];
            lifetimeplotter(ImagetoPlot, folder, AlphatoPlot, lifetimeLow, lifetimeHigh)
            else
            AlphaData = squeeze(AlphaData(a,:,:));
            AlphatoPlot = AlphaScalefactor*squeeze(AlphaData)/OverallNormalisedAlphaFactor;
            folder = [filePath,analysisType, '\LifetimeNormalised_', num2str(CurrentWavelength),'nm','\r', num2str(row),'_c', num2str(colum), '_Lifetime ', num2str(CurrentWavelength),'nm.tif'];
            lifetimeplotter(ImagetoPlot, folder, AlphatoPlot, lifetimeLow, lifetimeHigh)
            end
        end
    end
end
else
end

%% Create Video 
% can run if "LifetimeAlphaData"  and "LifetimeImageData" are loaded  along 
% with runing the  initialiation section of the script
        imageLifetimeData = lifetimeDatatoAnalyse{row,colum};
        imageAlphaData = lifetimeDatatoAnalyse{row,colum};
    
if createVideo ==1
load cmap
% setup a video file to populate
filename = fullfile(newFolderLifetimeData, '\processed_video.avi');
compressedVideo = VideoWriter(filename); 
%set the final video frame rate
compressedVideo.FrameRate = frameRate;
compressedVideo.Quality = videoQuality;
% open the video file ready to start adding frames
open(compressedVideo);

NormalisedAlphaData=[];
for z = firstSensorPixel:pixeldivider:lastSensorPixel
    row = 0;
    for l = 1:numberofRows
        row = row + 1;
        colum = 0;
        
        for k = 1:numberofColums
            colum = colum + 1;
            imageAlphaData = alphaDatatoAnalyse{row,colum};
            AlphaDataWavelength =imageAlphaData(z,:,:);
            normalisationValue= max(max(AlphaDataWavelength));
            NormalisedAlphaData(z,imageNumber) = normalisationValue;
        end
    end
end

    a=0;
for z = firstSensorPixel:pixeldivider:lastSensorPixel
    row = 0;
    a=a+1;
    CurrentWavelength = round(z*0.5468 + 500);
    OverallNormalisedAlphaFactor = max(NormalisedAlphaData(z,:));
    for l = 1:numberofRows
        row = row + 1;
        colum = 0;
        
        for k = 1:numberofColums
            colum = colum + 1;
            imageLifetimeData = lifetimeDatatoAnalyse{row,colum};
            ImagetoPlot =squeeze(imageLifetimeData(z,:,:));
            AlphaData = alphaDatatoAnalyse{row,colum};
            
            if oneMinusAlpha == 1
            AlphaData = squeeze(AlphaData(z,:,:));
            AlphatoPlot = AlphaScalefactor*(1-(AlphaData/OverallNormalisedAlphaFactor));
            [compressedVideo] =  lifetimeplotter_video(ImagetoPlot,  AlphatoPlot, lifetimeLow, lifetimeHigh, compressedVideo,CurrentWavelength, cmap);
            else
            AlphaData = squeeze(AlphaData(z,:,:));
            AlphatoPlot = AlphaScalefactor*AlphaData/max(max(AlphaData));
            %AlphatoPlot = AlphaScalefactor*AlphaData/OverallNormalisedAlphaFactor;
            [compressedVideo] =  lifetimeplotter_video(ImagetoPlot,  AlphatoPlot, lifetimeLow, lifetimeHigh, compressedVideo,CurrentWavelength,  cmap);
            end
        end
    end
end
%close the video file
close(compressedVideo);
else
end

%%
 % Impllement saving in non cell format?
% lifetimeImageDatatoSave = cell2mat(lifetimeImageData);
% lifetimeImageDatatoSave = permute(reshape(lifetimeImageData,[frame_size_x, numberOfwavelengthstoPlot, frame_size_x]),[2 1 3]);
% lifetimeAlphaDatatoSave = cell2mat(lifetimeAlphaData);
% lifetimeAlphaDatatoSave = permute(reshape(lifetimeAlphaData,[frame_size_x, numberOfwavelengthstoPlot, frame_size_x]),[2 1 3]);

save([newFolderLifetimeData,'\LifetimeImageData.mat'],'lifetimeImageData')
save([newFolderLifetimeData,'\LifetimeAlphaData.mat'],'lifetimeImageData')
%clear all;
            
