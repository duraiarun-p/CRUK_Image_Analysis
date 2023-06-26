%% Spectral Analysis of LFS and SEXP Fitting
clc;clear;close all;
%%
currentFolder = pwd;
%%
%  load('C:\Users\THT_CRUK\Documents\MATLAB\Test_Data\512Band_data\HistMode_no_pixel_binning\5_3x3_Row_1_col_1\workspace.frame_1.mat');
filePath='C:\Users\THT_CRUK\Documents\MATLAB\Test_Data\512Band_data\HistMode_no_pixel_binning\';
%% Spectrum extraction

Int_Prof=myread_file(filePath,currentFolder,[1,14],25);
%% Spectrum array arrangement
siz=size(Int_Prof);
siz1=size(Int_Prof{1,1});
F_sp_all=zeros(siz1(1),siz(1));

for tile=1:siz(1)
F_sp_all(:,tile)=Int_Prof{tile,3};
end
figure(29);
mesh(F_sp_all);
ylabel('Spectrum index');
xlabel('Tile index')
zlabel('Intensity');
title('Intensity spectrum across tiles @ t=T_{1/2}')
%% Spectrum conditioning
for tile=1:siz(1)
F_sp_all(:,tile)=Int_Prof{tile,2};
end
figure(30);
mesh(F_sp_all);
ylabel('Spectrum index');
xlabel('Tile index')
zlabel('Intensity');
title('Intensity spectrum across time')

F_sp=mean(F_sp_all,2);
F_sp_s=smooth(F_sp,10);
figure(31),
plot(F_sp),
hold all;
plot(F_sp_s);
hold off;
xlabel('Spectrum index');
ylabel('Intensity');
legend({'raw','smoothened'});
title('Intensity spectrum');
%% Peak and valley detection
[lambda_pks,lambda_pks_locs] = findpeaks(F_sp_s);
[lambda_vly,lambda_vly_locs] = findpeaks(-F_sp_s);
lambda_vly=-lambda_vly;
nopeaks=3;

lambda_pks(nopeaks+1:end)=[];
lambda_pks_locs(nopeaks+1:end)=[];
lambda_vly(nopeaks+1:end)=[];
lambda_vly_locs(nopeaks+1:end)=[];

figure(32),
plot(F_sp_s),
hold all;
plot(lambda_pks_locs,lambda_pks,'x');
plot(lambda_vly_locs,lambda_vly,'o');
hold off;
xlabel('Spectrum index');
ylabel('Intensity');
legend({'data','peaks','valleys'});
title('Intensity spectrum');

%% Aggregation of spectral indices
sp_indx = union(lambda_pks_locs,lambda_vly_locs);
F_sp_indx=F_sp_s(sp_indx);

figure(33),
plot(F_sp_s),
hold all;
plot(sp_indx,F_sp_indx,'x');
% plot(lambda_vly_locs,lambda_vly,'o');
hold off;
xlabel('Spectrum index');
ylabel('Intensity');
legend({'data','peaks'});
title('Intensity spectrum');

%% Automatic spectral index selection

% lambda_pt1=lambda_pks_locs(1);
% lambda_pt1=lambda_vly_locs(1);
lambda_pt1=1;
lambda_pt2=lambda_vly_locs(end);
numberoflambdas=20;
lambdas = round(linspace(lambda_pt1,lambda_pt2,numberoflambdas));
F_sp_lambdas=F_sp_s(lambdas);

figure(34),
plot(F_sp_s),
hold all;
plot(lambdas,F_sp_lambdas,'x');
% plot(lambda_vly_locs,lambda_vly,'o');
hold off;
xlabel('Spectrum index');
ylabel('Intensity');
legend({'data','spectral lambda'});
title('Intensity spectrum');

    %% functions for support
function Int_Prof=myread_file(filePath,currentFolder,binToFit,fignum)
    file_list=dir(filePath);
first_file_path=strcat(file_list(end).folder,'\',file_list(end).name);
first_file_mat_paths=dir(first_file_path);


% number of folders by name, before data starts, remember "." and ".."
% this is for folders within the "HistMode_no_pixel_binning" folder
% the folder selected in the popup should be one level above this folder.
% numberofNondataFolders = 3;
% Previously hardcoded now changed with this subroutine
file_list_names={file_list.name}; % Get file names into cell array
data_file_index=find(contains(file_list_names,'Row')); % data file indices
numberofNondataFolders=abs(length(data_file_index)-length(file_list_names));


% first_file_mat_file=load([first_file_path,'\','workspace.frame_1.mat']); % To continue with data acquistion convention
% Sub routine to extract number of rows and column
mat_file_name=file_list(end).name;
newStr = split(mat_file_name,'_');
newstr=str2double(newStr);
rows_cols=newstr(~(isnan(newstr)));
numberofColums=rows_cols(end);
rows_cols(end)=[];
numberofRows=rows_cols(end);

row = 0;
ind=1;
row_ind=zeros(numberofRows*numberofColums,1);
col_ind=zeros(numberofRows*numberofColums,1);
for r = 1:numberofRows
    row = row + 1;

    colum = 0;
    for k = 1:numberofColums
        colum = colum + 1;
        row_ind(ind)=row;
        col_ind(ind)=colum;
        ind=ind+1;
    end
end

all_files = dir(filePath);
all_files = struct2table(all_files);
all_files = sortrows(all_files, 'name');
all_files = table2struct(all_files);

Int_Prof=cell(ind-1,3);

% FT_sp=zeros(,ind-1);

 for file_ind=1:(ind-1)
        row=row_ind(file_ind);
        colum=col_ind(file_ind);
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
        load('workspace.frame_1.mat','bins_array_3')
        %return to matlab scripts directory
        cd(currentFolder)
        binToFit1 = binToFit(1);
        binToFit2 = binToFit(2);
        bins_array_3_a=bins_array_3(:,binToFit1:binToFit2,:,:);
        [VCm1,VCm,VC1]=process_bin(bins_array_3_a);
        disp('Producing Intensity Image');
        Int_Prof{file_ind,1}=VC1;
        Int_Prof{file_ind,2}=VCm;
        Int_Prof{file_ind,3}=VCm1;


        figure(fignum),
        title('Intensity spectrum across time')
        subplot(numberofRows,numberofColums,file_ind),
        plot(VCm);
        xlabel('Spectrum index');
        ylabel('Intensity');
        subtitle(['tile=',num2str(file_ind)])
        hold all;
        figure(fignum+1),
        title('Intensity spectrum @ t=T_{1/2}');
        subplot(numberofRows,numberofColums,file_ind),
        plot(VCm1);
        xlabel('Spectrum index');
        ylabel('Intensity');
        subtitle(['tile=',num2str(file_ind)])
        hold all;
        


 end
 hold off;
 
end

function [VCm1,VCm,VC1]=process_bin(bins_array_3)
 siz=size(bins_array_3);
 
%  lambda=1;
VC=zeros(siz(1),siz(2));
VC1=zeros(siz(1),siz(2));

    for t=1:siz(2)
        for lambda=1:siz(1)
        V=bins_array_3(lambda,t,:,:);
        VC(lambda,t)=sum(V(:));
        end
        VC1(:,t)=smooth(VC(:,t),15);
    end

    VCm=mean(VC1,2);
    VCm1=VC1(:,round(siz(2)*0.25));
end