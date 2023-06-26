%%test analysis script
%% SETUP PERAMITERS
% number of folders by name, before data starts, remember "." and ".."
% this is for folders within the "HistMode_no_pixel_binning" folder
% the folder selected in the popup should be one level above this folder.
numberofNondataFolders = 3;

%setnumberofrows/colums - to match data as recorded, this will effect the
%number of files read in and the naming of output files
numberofRows = 4; %this is the number of rows OR just change to the number of samples to analyse
numberofColums = 4;

% numberofRows = 3; %this is the number of rows OR just change to the number of samples to analyse
% numberofColums = 3;

n_bins = 16;
n_spectrum = 8;
% n_spectrum = 8;
frame_size = 512;
% frame_size = 256;



%Set name for output toplevel folder where data is saved

% analysisType = '/20221202_PutSessionNameHere'; % CHANGE THIS FOR EACH DIFFERENT ANALYSIS
                                                            
AlphaMask = 1; % set to 0 to plot with no Aplha masking applied - TO BE IMPLEMENTED
AlphaScalefactor = 2.5; % Scaling factor for alpha mask (contrast level)
oneMinusAlpha = 0; %set to 1 for a 1-Alpha plot

%scaling for Intesnity normalisation, increase if needed (if using more
%than 1 image in a single run
scalingFactorIntensity = 1.2;

%set the wavelength range to look over (need to convet drom wavelength 1 =
%500 nm, 512 = 780 nm)
firstSensorPixel = 1;
lastSensorPixel = 8;
numberOfwavelengthstoPlot = 8; % number of wavelengths to fit beteen 500 and 720nm, evenly spaced

% set number of spectral pixels for moving sum, increase if mean Tau data noisy
spectral_pixel_span = 64; 

% set threashold for when lifetime is set to NaN, based on peak bin fitted
count_threshold = 200;
mask_threshold = count_threshold;

% if you want to only want to create the lifetime data cubes:
% numberOfwavelengthstoPlot = 512,firstSensorPixel = 1 ,lastSensorPixel =
% 512 then set all the ploting / video options below to 0

plotImages = 0; % set to 1 to plot and save lifetime images set to 0 to simply save data
plotNormalisedImages = 0; % set to 1 to plot and save normalised lifetime images set to 0 to simply save data
createVideo = 0; % set to 1 to create a video of the computed lifetime images with histograms
videoQuality = 60; % set between 1 and 100 if output video too large
frameRate=20; % 45-60 works well for full spectral

%select 1st and last bin for fitting ALWAYS CHECK THESE LOOK GOOD FOR YOUR
%DATA - USE 
binToFit1 = 10;
binToFit2 = 14;

%set hitmode
histMode = 3;



%peramiters for plotting=
bin = 2; % for alpha mask
sample = 'test';

lifetimeLow = 1; % for stained, 0.7,  1.5 for unstained/fresh, MHorrick 1
lifetimeHigh = 2.5; % for stained, 1.7,  2.8 for unstained/fresh, MHorrricks 2
      

% % Load file path and find number of folders - 1 level deap to workspaces!!
currentFolder = pwd;
filePath = uigetdir; % User Interaction Dialogue

% newFolderIntesnity = [filePath, analysisType, '/Intensity'];
% mkdir(newFolderIntesnity);
% newFolderIntesnityNormalised = [filePath, analysisType, '/Intensity/Normalised'];
% mkdir(newFolderIntesnityNormalised);
% newFolderHistograms = [filePath, analysisType, '/Histograms'];
% mkdir(newFolderHistograms);
% newFolderMeanTau = [filePath, analysisType, '/Histograms/meanTau'];
% mkdir(newFolderMeanTau);
% newFolderLifetimeData = [filePath, analysisType, '/Lifetime_Data'];
% mkdir(newFolderLifetimeData);

% Variables for saving the plots
pixeldivider = round((lastSensorPixel-firstSensorPixel)/numberOfwavelengthstoPlot);
lastSensorPixel = pixeldivider*numberOfwavelengthstoPlot;


Wavelength = 0;
for i = firstSensorPixel:pixeldivider:lastSensorPixel
    Wavelength = Wavelength +1;
    Wave  = round(i*(600-470)/8+ 470);
%     
%     if plotImages == 1
%     newFolderLifetimeImages = [filePath, analysisType,  '/Lifetime_', num2str(Wave), 'nm'];
%     mkdir(newFolderLifetimeImages);
%     
%     newFolderHistogramsData = [filePath, analysisType, '/Histograms/', num2str(Wave),'nm'];
%     mkdir(newFolderHistogramsData);
%     end
%     
%     if plotNormalisedImages == 1
%     newFolderLifetimeImages = [filePath, analysisType,  '/LifetimeNormalised_', num2str(Wave), 'nm'];
%     mkdir(newFolderLifetimeImages);
%     end 
%
end

% pause(0.1)
% 
% all_files = dir([filePath, '/HistMode_full_8bands_pixel_binning_inFW/']);
all_files = dir(filePath);
all_files = struct2table(all_files);
all_files = sortrows(all_files, 'name');
all_files = table2struct(all_files);



%%
%loop through images in date order, assumes the data was recorded in row by
%row with the same starting point, data saved to the corresponding folder,
%data workspaces must be only 1 level deap from the main folder
% allIntensityImages={};
% lifetimeImageData={};
% lifetimeAlphaData={};

allIntensityImages=cell(1,numberofRows*numberofColums);
lifetimeImageData=cell(1,numberofRows*numberofColums);
lifetimeAlphaData=cell(1,numberofRows*numberofColums);

row = 0;

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
        %move to and load worspace from 1st subfolder
%         currDir = [filePath,'/HistMode_full_8bands_pixel_binning_inFW/', all_files(fileNumber).name];
        currDir = [filePath,'/',all_files(fileNumber).name];
        cd(currDir)
        disp('Loading workspace for folder:')
        disp(all_files(fileNumber).name)
        load('workspace.frame_1.mat')
        %return to matlab scripts directory
        cd(currentFolder)

        bins_array_3 = permute(bins_array_3, [4 3 1 2]); % This is the place that check for array dimension and changes if needed
        
        disp('Producing Intensity Image')

        % Produce and save intensity images
%         [intensity_image] = Intensity_Image_Summation(bins_array_3, frame_size_x);
        intensity_image = sum(sum(bins_array_3, 1), 2);
        climit = 'auto';
%         plotter(intensity_image, newFolderIntesnity, row, colum, climit)
        
        allIntensityImages{row+colum-1 + ((row-1)*(numberofColums-1))} = intensity_image;
               
        %Calculate wavelength axis
%         [wavelengths,wavenumbers] = Wavelength_Calculator();
        
        disp('Performing Lifetime Calculations')
        % do lifetime fit calculations
        [parameters_cpu, selected_data_for_subtraction, bins_array_movsum_selected_reshaped] = test_LM_fitting_linear_gw(bins_array_3, histMode, spectral_pixel_span, binToFit1, binToFit2, frame_size, n_bins, n_spectrum);
        
        if plotImages == 1
            disp('Producing Lifetime Images and Plots for wavelength:')
        else
            disp('Producing Datacubes')
        end
        % Produce lifetime plots and histograms for various wavelengths
        
        numberofbins = size(selected_data_for_subtraction(:,1),1);
        selected_data_for_subtractionPeakbin = selected_data_for_subtraction(numberofbins,:);
        mask = selected_data_for_subtractionPeakbin;
        mask(mask<count_threshold)=0;
        mask(mask>count_threshold)=1;
        parameters_cpu(2,:) = parameters_cpu(2,:).*mask;
        tauLeastSquaresCPU = parameters_cpu(2,:); 
        tauLeastSquaresCPU(tauLeastSquaresCPU>5)=0;
        tauLeastSquaresReshaped = reshape(tauLeastSquaresCPU, [n_spectrum frame_size_x frame_size_x]);
        AlphaDataAll = reshape(selected_data_for_subtractionPeakbin, [n_spectrum frame_size_x frame_size_x]);
        
        
%         wavelengthnumber = 0;
%         for i = firstSensorPixel:pixeldivider:lastSensorPixel
%             
%            
%             wavelengthnumber = wavelengthnumber+1;
%             spectral_pixel = i;
%             
%             if plotImages == 1
%             Currentwavelength = i*0.5468 + 500;
%             disp(Currentwavelength)
% %             Lifetime_Image_Creation(spectral_pixel, bin, mask_threshold, sample, count_threshold, selected_data_for_subtraction, parameters_cpu, bins_array_movsum_selected_reshaped, wavelengths, lifetimeLow, lifetimeHigh, filePath, row, colum,analysisType, AlphaScalefactor, oneMinusAlpha);
%             end
%                 
%             if plotImages == 0
%             tauLeastSquaresReshapedDisplayFrame = reshape(tauLeastSquaresReshaped(spectral_pixel,:,:),[frame_size frame_size]);
%             array_movsum_selected = reshape(bins_array_movsum_selected_reshaped, size(bins_array_movsum_selected_reshaped, 2), size(bins_array_movsum_selected_reshaped, 1), size(bins_array_movsum_selected_reshaped, 3));
%             bins_array_alpha = reshape(array_movsum_selected(spectral_pixel, bin, :),[frame_size frame_size]);
%             end
% 
%         end

        lifetimeImageData{imageNumber} = tauLeastSquaresReshaped;
        lifetimeAlphaData{imageNumber} = AlphaDataAll;

    end
           


end

%% plot normalized intensity image - can be run independently if "allIntensityImages"
% is loaded and and the parameter initialisation section of the script is
% run
if plotNormalisedImages == 0
    AllnormalisationValue = zeros(1,numberofRows*numberofColums);
    disp('Creating Normalised Intesnity Images');
    row = 0; 
    for l = 1:numberofRows
        row = row + 1;
        colum = 0;
        for k = 1:numberofColums
            colum = colum + 1;
            imageNumber = row+colum-1 + ((row-1)*(numberofColums-1));
            ImagetoData =allIntensityImages{imageNumber};
            normalisationValue= max(max(ImagetoData));
            AllnormalisationValue(imageNumber) = normalisationValue;
        end
    end
    overallNormalisationValue = max(AllnormalisationValue);
    overallNormalisationValue = overallNormalisationValue/scalingFactorIntensity; 
    AllIntensityImagesNormalised = cell(1,numberofRows*numberofColums);
    row = 0; 
    for l = 1:numberofRows
        row = row + 1;
        colum = 0;
        for k = 1:numberofColums
            colum = colum + 1;
            imageNumber = row+colum-1 + ((row-1)*(numberofColums-1));
            ImagetoPlot =allIntensityImages{imageNumber};
            IntensityImagesNormalised=ImagetoPlot/overallNormalisationValue;
            AllIntensityImagesNormalised{imageNumber}= IntensityImagesNormalised;
            climit = [0 1];
%             plotter(IntensityImagesNormalised, newFolderIntesnityNormalised, row, colum, climit);
        end
    end

%     save([newFolderIntesnityNormalised,'/AllIntensityData.mat'],'allIntensityImages')
%     save([newFolderIntesnityNormalised,'/AllIntensityImagesNormalised.mat'],'AllIntensityImagesNormalised')
    else
end
%% Plot global normalized Alpha Lifetime plots, - can be run independently 
% if "LifetimeAlphaData"  and "LifetimeImageData" are loaded  along 
% with runing the  initialiation section of the script

if plotNormalisedImages == 0
    disp('Creating Normalised Lifetime Images');
%     NormalisedAlphaData=[];
    NormalisedAlphaData=zeros(Wavelength,numberofRows*numberofColums);
    for z = 1:Wavelength
        row = 0;
        for l = 1:numberofRows
            row = row + 1;
            colum = 0;

            for k = 1:numberofColums
                colum = colum + 1;
                imageNumber = row+colum-1 + ((row-1)*(numberofColums-1));
                imageAlphaData = lifetimeAlphaData{imageNumber};
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
%                 folder = [filePath,analysisType, '/LifetimeNormalised_', num2str(CurrentWavelength),'nm','\r', num2str(row),'_c', num2str(colum), '_Lifetime ', num2str(CurrentWavelength),'nm.tif'];
%                 lifetimeplotter(ImagetoPlot, folder, AlphatoPlot, lifetimeLow, lifetimeHigh)
                else
                AlphaData = squeeze(AlphaData(a,:,:));
                AlphatoPlot = AlphaScalefactor*squeeze(AlphaData)/OverallNormalisedAlphaFactor;
%                 folder = [filePath,analysisType, '/LifetimeNormalised_', num2str(CurrentWavelength),'nm','\r', num2str(row),'_c', num2str(colum), '_Lifetime ', num2str(CurrentWavelength),'nm.tif'];
%                 lifetimeplotter(ImagetoPlot, folder, AlphatoPlot, lifetimeLow, lifetimeHigh)
                end
            end
        end
    end
    else
end

%% Create Video 
% can run if "LifetimeAlphaData"  and "LifetimeImageData" are loaded  along 
% with runing the  initialiation section of the script

    
if createVideo ==1
    disp('Creating Video');
    load cmap
    % setup a video file to populate
    filename = fullfile(newFolderLifetimeData, '/processed_video_modifiedcolors1.avi');
    compressedVideo = VideoWriter(filename); 
    %set the final video frame rate
    compressedVideo.FrameRate = frameRate;
    compressedVideo.Quality = videoQuality;

    % open the video file ready to start adding frames
    open(compressedVideo);

%     NormalisedAlphaData=[];
    NormalisedAlphaData=zeros(Wavelength,numberofRows*numberofColums);
    for z = firstSensorPixel:pixeldivider:lastSensorPixel
        row = 0;
        for l = 1:numberofRows
            row = row + 1;
            colum = 0;

            for k = 1:numberofColums
                colum = colum + 1;
                imageNumber = row+colum-1 + ((row-1)*(numberofColums-1));
                imageAlphaData = lifetimeAlphaData{imageNumber};
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
                imageNumber = row+colum-1 + ((row-1)*(numberofColums-1));
                ImageData =lifetimeImageData{imageNumber};
                ImagetoPlot =squeeze(ImageData(z,:,:));
                AlphaData = lifetimeAlphaData{imageNumber};

                if oneMinusAlpha == 1
                AlphaData = squeeze(AlphaData(z,:,:));
                AlphatoPlot = AlphaScalefactor*(1-(AlphaData/OverallNormalisedAlphaFactor));
                [compressedVideo] =  lifetimeplotter_video(ImagetoPlot,  AlphatoPlot, lifetimeLow, lifetimeHigh, compressedVideo,CurrentWavelength, bins_array_movsum_selected_reshaped, bin, mask_threshold, z, cmap);
                else
                AlphaData = squeeze(AlphaData(z,:,:));
                AlphatoPlot = AlphaScalefactor*AlphaData/max(max(AlphaData));
                [compressedVideo] =  lifetimeplotter_video(ImagetoPlot,  AlphatoPlot, lifetimeLow, lifetimeHigh, compressedVideo,CurrentWavelength, bins_array_movsum_selected_reshaped, bin, mask_threshold, z, cmap);
                end
            end
        end
    end
    %close the video file
    close(compressedVideo);
    else
end
% 
% %
%  % Impllement saving in non cell format?
% % lifetimeImageDatatoSave = cell2mat(lifetimeImageData);
% % lifetimeImageDatatoSave = permute(reshape(lifetimeImageData,[frame_size_x, numberOfwavelengthstoPlot, frame_size_x]),[2 1 3]);
% % lifetimeAlphaDatatoSave = cell2mat(lifetimeAlphaData);
% % lifetimeAlphaDatatoSave = permute(reshape(lifetimeAlphaData,[frame_size_x, numberOfwavelengthstoPlot, frame_size_x]),[2 1 3]);
% disp('Saving Datacubes');
% save([newFolderLifetimeData,'/LifetimeImageData.mat'],'lifetimeImageData')
% save([newFolderLifetimeData,'/LifetimeAlphaData.mat'],'lifetimeAlphaData')

%% create Meta Data File
% xgalvo_step_size = 35;
% firstWavelength = 0.5468*firstSensorPixel + 500;
% lastWavelength = 0.5468*lastSensorPixel + 500;
% AnalysedfolderName = split(filePath,"/");
% metaDataFolderName = strcat('/metaData_', string(AnalysedfolderName(2)), '.csv');
% metaData = {};
% metaData{1}  = ["Folder Analysed " ,  filePath];
% metaData{2}  = ["Hist mode " ,  num2str(HIST_MODE)];
% metaData{3}  = ["Pstop" , num2str(CODE_PSTOP)];
% metaData{4}  = ["Frame Size " , num2str(frame_size_x)];
% metaData{5}  = ["Step Size " , num2str(xgalvo_step_size)];
% metaData{6}  = ["Exposure Time " , num2str(exposure_time_us)];
% metaData{7}  = ["Bin1 for Fitting" , num2str(binToFit1)];
% metaData{8}  = ["Bin2 for Fitting " , num2str(binToFit2)];
% metaData{9}  = ["Alpha Mask Enabled " , num2str(AlphaMask)];
% metaData{10} = ["Alpha Scale Factor " , num2str(AlphaScalefactor)];
% metaData{11} = ["1-Alpha Enabled " , num2str(oneMinusAlpha)];
% metaData{12} = ["Count Threashold " , num2str(count_threshold)];
% metaData{13} = ["Number of Wavelengths Analysed " , num2str(numberOfwavelengthstoPlot)];
% metaData{14} = ["Starting Wavelength " , num2str(firstWavelength)];
% metaData{15} = ["Last Wavelength " , num2str(lastWavelength)];
% metaData{16} = ["Short Lifetime for Plots " , num2str(lifetimeLow)];
% metaData{17} = ["Long Lifetime for Plots " , num2str(lifetimeHigh)];
% metaData{18} = ["Vidio Frame Rate " , num2str(frameRate)];
% metaData{19} = ["Vidio Compression (%) " , num2str(videoQuality)];
% metaData{19} = ["Moving Average Size " , num2str(spectral_pixel_span)];
% metaData = splitvars(cell2table(metaData'));
% metaData.Properties.VariableNames = {'Variable' 'Value'};
% writetable(metaData,strcat(filePath, analysisType , metaDataFolderName));

% pause (1);

