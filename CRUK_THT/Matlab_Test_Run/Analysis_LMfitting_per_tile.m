function [allIntensityImages,lifetimeImageData,lifetimeAlphaData]=Analysis_LMfitting_per_tile(bins_array_3,binToFit,histMode,frame_size)
% Temporal resolution
n_bins = 16;



% % Load file path and find number of folders - 1 level deap to workspaces!!
% currentFolder = pwd;
% filePath = uigetdir; % User Interaction Dialogue


%setnumberofrows/colums - to match data as recorded, this will effect the
%number of files read in and the naming of output files
% numberofRows = 4; %this is the number of rows OR just change to the number of samples to analyse
% numberofColums = 4;
% Previously hardcoded now, directly accessed from data folder using the
% following subroutine

% Extracting bin array parameters directly from the data
% file_list=dir(filePath);
% first_file_path=strcat(file_list(end).folder,'\',file_list(end).name);
% first_file_mat_paths=dir(first_file_path);


% number of folders by name, before data starts, remember "." and ".."
% this is for folders within the "HistMode_no_pixel_binning" folder
% the folder selected in the popup should be one level above this folder.
% numberofNondataFolders = 3;

% Previously hardcoded now changed with this subroutine
% file_list_names={file_list.name}; % Get file names into cell array
% data_file_index=find(contains(file_list_names,'Row')); % data file indices
% numberofNondataFolders=abs(length(data_file_index)-length(file_list_names));


% first_file_mat_file=load([first_file_path,'\','workspace.frame_1.mat']); % To continue with data acquistion convention
% % Sub routine to extract number of rows and column
% mat_file_name=file_list(end).name;
% newStr = split(mat_file_name,'_');
% newstr=str2double(newStr);
% rows_cols=newstr(~(isnan(newstr)));
% numberofColums=rows_cols(end);
% rows_cols(end)=[];
% numberofRows=rows_cols(end);
% numberofColums=rows_cols(2);

%set hitmode
% histMode = first_file_mat_file.HIST_MODE;

% Frame size
% frame_size = first_file_mat_file.frame_size_x;

% Extraction of bin_array_3 dimension's info
siz=size(bins_array_3);

% spatial_dimension_ind=find(siz==frame_size); % to permutate the array in the right order and maintain consistency
temporal_dimension_ind=find(siz==n_bins);
% remain_ind=union(temporal_dimension_ind,spatial_dimension_ind);
% spectral_dimension_ind=find(siz);
% spectral_dimension_ind(remain_ind)=[];


% Simple way of indexing
spatial_dimension_ind=[temporal_dimension_ind+1,temporal_dimension_ind+2];
spectral_dimension_ind=temporal_dimension_ind-1;


% Spectral Information
n_spectrum = siz(spectral_dimension_ind);
% line for automatic selection - spectral dimension based array slicing
% n_spectrum=length(lambdas);

% set number of spectral pixels for moving sum, increase if mean Tau data noisy
spectral_pixel_span = 64; 

% set threashold for when lifetime is set to NaN, based on peak bin fitted
count_threshold = 200;

binToFit1 = binToFit(1);
binToFit2 = binToFit(2);
% binToFit1 = 10;
% binToFit2 = 14;

%%

% allIntensityImages=cell(1,numberofRows*numberofColums);
% lifetimeImageData=cell(1,numberofRows*numberofColums);
% lifetimeAlphaData=cell(1,numberofRows*numberofColums);

% row = 0;




        bins_array_3 = permute(bins_array_3, [spectral_dimension_ind temporal_dimension_ind spatial_dimension_ind(1) spatial_dimension_ind(2)]); 
        disp('Producing Intensity Image')

        % Produce and save intensity images
%         [intensity_image] = Intensity_Image_Summation(bins_array_3, frame_size_x);
        intensity_image = sum(sum(bins_array_3, spectral_dimension_ind), temporal_dimension_ind);
%         climit = 'auto';
%         plotter(intensity_image, newFolderIntesnity, row, colum, climit)
        
        allIntensityImages = intensity_image;
               
        %Calculate wavelength axis
%         [wavelengths,wavenumbers] = Wavelength_Calculator();
        
        disp('Performing Lifetime Calculations')
        % do lifetime fit calculations
        [parameters_cpu, selected_data_for_subtraction, bins_array_movsum_selected_reshaped] = test_LM_fitting_linear_gw(bins_array_3, histMode, spectral_pixel_span, binToFit1, binToFit2, frame_size, n_bins, n_spectrum);
        
        % Produce lifetime plots and histograms for various wavelengths
        
        numberofbins = size(selected_data_for_subtraction(:,1),1);
        selected_data_for_subtractionPeakbin = selected_data_for_subtraction(numberofbins,:);
        mask = selected_data_for_subtractionPeakbin;
        mask(mask<count_threshold)=0;
        mask(mask>count_threshold)=1;
        parameters_cpu(2,:) = parameters_cpu(2,:).*mask;
        tauLeastSquaresCPU = parameters_cpu(2,:); 
        tauLeastSquaresCPU(tauLeastSquaresCPU>5)=0;
        tauLeastSquaresReshaped = reshape(tauLeastSquaresCPU, [n_spectrum frame_size frame_size]);
        AlphaDataAll = reshape(selected_data_for_subtractionPeakbin, [n_spectrum frame_size frame_size]);
        
                lifetimeImageData = tauLeastSquaresReshaped;
        lifetimeAlphaData = AlphaDataAll;

%     end



end