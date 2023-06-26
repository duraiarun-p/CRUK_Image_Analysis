%% Spectral Analysis of LFS and SEXP Fitting
clc;clear;close all;
%%
currentFolder = pwd;
%%
%  load('C:\Users\THT_CRUK\Documents\MATLAB\Test_Data\512Band_data\HistMode_no_pixel_binning\5_3x3_Row_1_col_1\workspace.frame_1.mat');
filePath='C:\Users\THT_CRUK\Documents\MATLAB\Test_Data\512Band_data\HistMode_no_pixel_binning\';
%% Spectrum extraction

Int_Prof=myread_file(filePath,currentFolder,24);

    %% functions for support
function Int_Prof=myread_file(filePath,currentFolder,fignum)
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

Int_Prof=cell(ind-1,4);

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
%         binToFit1 = binToFit(1);
%         binToFit2 = binToFit(2);
%         bins_array_3_a=bins_array_3(:,binToFit1:binToFit2,:,:);
%         [VCm1,VCm,VC1]=process_bin(bins_array_3);
%         disp('Producing Intensity Image');
%         Int_Prof{file_ind,1}=VC1;
%         Int_Prof{file_ind,2}=VCm;
%         Int_Prof{file_ind,3}=VCm1;
%         Int_Prof{file_ind,4}=bins_array_3;
% 
% 
%         figure(fignum),
%         title('Intensity spectrum across time')
%         subplot(numberofRows,numberofColums,file_ind),
%         plot(VCm);
%         xlabel('Spectrum index');
%         ylabel('Intensity');
%         subtitle(['tile=',num2str(file_ind)])
%         hold all;
%         figure(fignum+1),
%         title('Intensity spectrum @ t=T_{1/2}');
%         subplot(numberofRows,numberofColums,file_ind),
%         plot(VCm1);
%         xlabel('Spectrum index');
%         ylabel('Intensity');
%         subtitle(['tile=',num2str(file_ind)])
%         hold all;

[spectrums,times,xs,ys]=size(bins_array_3);

acc_arr1=zeros(spectrums,times);
acc_arr1L=zeros(spectrums,times);
acc_arr2L=zeros(spectrums,times);
acc_arr3L=zeros(spectrums,times);
acc_arr4L=zeros(spectrums,times);
acc_arr5L=zeros(spectrums,times);

for spec_1=1:spectrums
    for time_1=1:times
        bin=squeeze(bins_array_3(spec_1,time_1,:,:));
        acc_arr1(spec_1,time_1)=sum(bin(:));
        acc_arr1L(spec_1,time_1)=bin(xs*0.25,ys*0.25);
        acc_arr2L(spec_1,time_1)=bin(xs*0.75,ys*0.25);
        acc_arr3L(spec_1,time_1)=bin(xs*0.75,ys*0.25);
        acc_arr4L(spec_1,time_1)=bin(xs*0.75,ys*0.75);
        acc_arr5L(spec_1,time_1)=bin(xs*0.5,ys*0.5);

    end
end

acc_arr1a=mean(acc_arr1);
acc_arr1b=mean(acc_arr1,2);

acc_arr1La=mean(acc_arr1L);
acc_arr1Lb=mean(acc_arr1L,2);

acc_arr2La=mean(acc_arr2L);
acc_arr2Lb=mean(acc_arr2L,2);

acc_arr3La=mean(acc_arr3L);
acc_arr3Lb=mean(acc_arr3L,2);

acc_arr4La=mean(acc_arr4L);
acc_arr4Lb=mean(acc_arr4L,2);

acc_arr5La=mean(acc_arr5L);
acc_arr5Lb=mean(acc_arr5L,2);

figure(fignum+1);
subplot(numberofRows,numberofColums,file_ind),
plot(acc_arr1a);
xlabel('time');
ylabel('intensity')
subtitle(['tile=',num2str(file_ind)])
title('Time vs Intensity Across Spatial Dimension');

figure(fignum+2),
subplot(numberofRows,numberofColums,file_ind),
plot(acc_arr1La);hold all;
plot(acc_arr2La);
plot(acc_arr3La);
plot(acc_arr4La);
plot(acc_arr5La);hold off;
legend({'1st quarter','2nd quarter','3rd quarter','4th quarter','Centre'},'location','Northwest');
xlabel('time');
ylabel('intensity')
subtitle(['tile=',num2str(file_ind)])
title('Time vs Intensity at specific coordinates');

figure(fignum+3);
subplot(numberofRows,numberofColums,file_ind),
plot(acc_arr1b);
xlabel('time');
ylabel('intensity')
subtitle(['tile=',num2str(file_ind)])
title('Intensity spectrum Across Spatial Dimension');

figure(fignum+4),
subplot(numberofRows,numberofColums,file_ind),
plot(acc_arr1Lb);hold all;
plot(acc_arr2Lb);
plot(acc_arr3Lb);
plot(acc_arr4Lb);
plot(acc_arr5Lb);hold off;
subtitle(['tile=',num2str(file_ind)])
legend({'1st quarter','2nd quarter','3rd quarter','4th quarter','Centre'});
xlabel('spectrum');
ylabel('intensity')
title('Intensity spectrum at specific coordinates');

figure(fignum+5),
subplot(numberofRows,numberofColums,file_ind),
plot(acc_arr1Lb(1:spectrums*0.5));hold all;
plot(acc_arr2Lb(1:spectrums*0.5));
plot(acc_arr3Lb(1:spectrums*0.5));
plot(acc_arr4Lb(1:spectrums*0.5));
plot(acc_arr5Lb(1:spectrums*0.5));hold off;
subtitle(['tile=',num2str(file_ind)])
legend({'1st quarter','2nd quarter','3rd quarter','4th quarter','Centre'});
xlabel('spectrum');
ylabel('intensity')
title('Intensity spectrum at specific coordinates');
        


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