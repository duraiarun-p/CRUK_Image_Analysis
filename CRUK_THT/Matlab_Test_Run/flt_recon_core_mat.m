function flt_recon_core_mat(filePath,currentFolder,numberoflambdas,binToFit)

outputdir=strcat(filePath,'/Mat_output2/');
if isfolder(outputdir)==0
    mkdir(outputdir);
end

%%

file_list=dir(filePath);
[~,ind]=sort({file_list.name});
file_list = file_list(ind);
% first_file_path=strcat(file_list(end).folder,'\',file_list(end).name);% Windows
% first_file_path=strcat(file_list(end).folder,'/',file_list(end).name);% Linux
% first_file_mat_paths=dir(first_file_path);
n_bins = 16;


% number of folders by name, before data starts, remember "." and ".."
% this is for folders within the "HistMode_no_pixel_binning" folder
% the folder selected in the popup should be one level above this folder.
% numberofNondataFolders = 3;
% Previously hardcoded now changed with this subroutine
file_list_names={file_list.name}; % Get file names into cell array
data_file_index=find(contains(file_list_names,'Row')); % data file indices
file_list_1=file_list(data_file_index);
file_list=file_list_1;
clear file_list_1
% first_file_path=strcat(file_list(end).folder,'\',file_list(end).name);% Windows
first_file_path=strcat(file_list(end).folder,'/',file_list(end).name);% Linux

numberofNondataFolders=abs(length(data_file_index)-length(file_list_names));


% first_file_mat_file=load([first_file_path,'\','workspace.frame_1.mat']); % To continue with data acquistion convention
first_file_mat_file=load([first_file_path,'/','workspace.frame_1.mat']); % Linux To continue with data acquistion convention
% Sub routine to extract number of rows and column
mat_file_name=file_list(end).name;
newStr = split(mat_file_name,'_');
newstr=str2double(newStr);
rows_cols=newstr(~(isnan(newstr)));
numberofColums=rows_cols(end);
rows_cols(end)=[];
numberofRows=rows_cols(end);
% numberofColums=rows_cols(2);

%set hitmode
histMode = first_file_mat_file.HIST_MODE;

% Frame size
frame_size = first_file_mat_file.frame_size_x;
frame_size_x = frame_size;

% Extraction of bin_array_3 dimension's info
siz=size(first_file_mat_file.bins_array_3);

lambdas = round(linspace(1,numberoflambdas,numberoflambdas));

spatial_dimension_ind=find(siz==frame_size); % to permutate the array in the right order and maintain consistency
temporal_dimension_ind=find(siz==n_bins);
remain_ind=union(temporal_dimension_ind,spatial_dimension_ind);
spectral_dimension_ind=find(siz);
spectral_dimension_ind(remain_ind)=[];

% Spectral Information
% n_spectrum = siz(spectral_dimension_ind);
n_spectrum=length(lambdas);

%Set name for output toplevel folder where data is saved

% analysisType = '/20221202_PutSessionNameHere'; % CHANGE THIS FOR EACH DIFFERENT ANALYSIS
                                                            
% AlphaMask = 1; % set to 0 to plot with no Aplha masking applied - TO BE IMPLEMENTED
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

% binToFit1 = binToFit(1);
% binToFit2 = binToFit(2);
% binToFit1 = 10;
% binToFit2 = 14;

%peramiters for plotting=
bin = 2; % for alpha mask
% sample = 'test';

lifetimeLow = 1; % for stained, 0.7,  1.5 for unstained/fresh, MHorrick 1
lifetimeHigh = 2.5; % for stained, 1.7,  2.8 for unstained/fresh, MHorrricks 2
      

allIntensityImages=cell(1,numberofRows*numberofColums);
lifetimeImageData=cell(1,numberofRows*numberofColums);
lifetimeAlphaData=cell(1,numberofRows*numberofColums);
time_LS=zeros(1,numberofRows*numberofColums);

% allIntensityImages_PEXP=cell(1,numberofRows*numberofColums);
% lifetimeImageData_PEXP=cell(1,numberofRows*numberofColums);
% lifetimeAlphaData_PEXP=cell(1,numberofRows*numberofColums);
% time_PEXP=zeros(1,numberofRows*numberofColums);
% 
% allIntensityImages_FEXP=cell(1,numberofRows*numberofColums);
% lifetimeImageData_FEXP=cell(1,numberofRows*numberofColums);
% lifetimeAlphaData_FEXP=cell(1,numberofRows*numberofColums);
% time_FEXP=zeros(1,numberofRows*numberofColums);

% row = 0;

% row = 0;
% ind=1;
% row_ind=zeros(numberofRows*numberofColums,1);
% col_ind=zeros(numberofRows*numberofColums,1);
% for r = 1:numberofRows
%     row = row + 1;
% 
%     colum = 0;
%     for k = 1:numberofColums
%         colum = colum + 1;
%         row_ind(ind)=row;
%         col_ind(ind)=colum;
%         ind=ind+1;
%     end
% end

% all_files = dir(filePath);
% file_list_names={all_files.name}; % Get file names into cell array
% data_file_index=find(contains(all_files,'Row')); % data file indices
% all_files(1:numberofNondataFolders)=[];

all_files=file_list;

all_files = struct2table(all_files);
all_files = sortrows(all_files, 'name');
all_files = table2struct(all_files);

all_files_len=length(all_files);
row_ind=zeros(all_files_len,1);
col_ind=zeros(all_files_len,1);
%%

% ind=2;
%%
tic;
for file_ind=1:all_files_len
        % row=row_ind(file_ind);
        % colum=col_ind(file_ind);
        % disp('row')
        % disp(row)
        % disp('column')
        % disp(colum)

        
       
        % fileNumber = row+colum-1 + ((row-1)*(numberofColums-1))+numberofNondataFolders;
        % imageNumber = row+colum-1 + ((row-1)*(numberofColums-1));
        % disp(fileNumber)
        % disp(imageNumber)
        stepp=split(all_files(file_ind).name,'_');
        row_ind(file_ind)=str2double(stepp{2,1});
        col_ind(file_ind)=str2double(stepp{4,1});

        %move to and load worspace from 1st subfolder
        currDir = [filePath,'/',all_files(file_ind).name];
        cd(currDir)
        disp(currDir);
        disp('Loading workspace for folder:')
        disp(all_files(file_ind).name)
        try
        disp('Loading')
        load('workspace.frame_1.mat')

        %return to matlab scripts directory
        cd(currentFolder)
        disp('Loaded')
        

%         bins_array_3 = permute(bins_array_3, [4 3 1 2]); % This is the place that check for array dimension and changes if needed
        % Instead of hardcoding the dimension, the dimension information
        % extracted initially will be used here for consistency
        % Extracting specific spectral responses using the lambdas variable

%         bins_array_3=bins_cell{file_ind,1}; % changed the col
        bins_array_3=bins_array_3(lambdas,:,:,:);
      disp('Fitting');
        % tic;
        [allIntensityImages1,lifetimeImageData1,lifetimeAlphaData1]=Analysis_LMfitting_per_tile(bins_array_3,binToFit,histMode,frame_size);
        time_LS(file_ind)=toc;
        disp("fitting complete - execution time: " + time_LS(file_ind) + " seconds");

        catch
            % continue;
        allIntensityImages1=zeros(frame_size_x,frame_size_x);
        lifetimeImageData1=zeros(n_spectrum,frame_size_x,frame_size_x);
        lifetimeAlphaData1=zeros(n_spectrum,frame_size_x,frame_size_x);
        disp("fitting was not complete");
        end
%         lifetimeImageData1=permute(lifetimeImageData1,[2 3 spectral_dimension_ind]);
        allIntensityImages1=squeeze(allIntensityImages1);
        allIntensityImages{file_ind} = allIntensityImages1;
        lifetimeImageData{file_ind} = lifetimeImageData1;
        lifetimeAlphaData{file_ind} = lifetimeAlphaData1;
        % time_LS(file_ind)=toc;
        
        

        matfilename=[outputdir,num2str(file_ind),'.mat'];
        % matfilename=[currentFolder,num2str(file_ind),'.mat'];
        imgfilename=[outputdir,num2str(file_ind),'.tiff'];
        imwrite(rescale(allIntensityImages1),imgfilename)

    % save(matfilename,'allIntensityImages1','lifetimeImageData1','lifetimeAlphaData1','-v7.3');
    % copyfile matfilename outputdir
%     end
end
matcorefilename=[outputdir,'core_all','.mat'];
% save(matcorefilename,'allIntensityImages','lifetimeImageData','lifetimeAlphaData','-v7.3');
end_time=toc;
disp((end_time)/60);
end