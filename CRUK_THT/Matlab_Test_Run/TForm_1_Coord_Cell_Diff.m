clc;clear;close all;
%%
% base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-2_20230215/Mat_output';
% % base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-9_20230222/Mat_output';
% base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-13_20230226/Mat_output';
% base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-3_Col-5_20230218/Mat_output2';
% base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/RT/Row-4_Col-1_20230214/Mat_output2';
% base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/RT/Row-6_Col-10_20230223/Mat_output2';


base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/Row-1_Col-9_20230222/FLT_IMG_DIR/Stitched';
%% Classification file extraction 
cd(base_dir)
files=dir(base_dir);
files_names=cell(length(files),1);
for ind = 1 :length(files)
files_names{ind,1}=files(ind).name;
end
file_index_classification_txt_ind=find(contains(files_names,'classification_QuPath.txt'));
file_index_tforms_ind=find(contains(files_names,'tforms'));

%% 
lines=readlines(files(file_index_classification_txt_ind).name);
load(files(file_index_tforms_ind).name)
%%
column_names=strsplit(lines(1),'\t');
lines(1)=[];
%%
% cell_centroid_x_ind=column_names.index("Centroid X px")
% cell_centroid_y_ind=column_names.index("Centroid Y px")
% cell_class=column_names.index("Class")
% cell_roi=column_names.index("ROI")
% cell_area=column_names.index("Cell: Area")
% cell_perimeter=column_names.index("Cell: Perimeter")
% cell_caliper_max=column_names.index("Cell: Max caliper")
% cell_caliper_min=column_names.index("Cell: Min caliper")

cell_items=cell(length(lines)-1,1);
listoflabels=cell(length(lines)-1,1);
class_grnd_trth=zeros(length(lines)-1,1);

cell_centroid_x_ind=find(column_names=="Centroid X px");
cell_centroid_y_ind=find(column_names=="Centroid Y px");
cell_class=find(column_names=="Class");
cell_roi=find(column_names=="ROI");
cell_area=find(column_names=="Cell: Area");
cell_perimeter=find(column_names=="Cell: Perimeter");
cell_caliper_max=find(column_names=="Cell: Max caliper");
cell_caliper_min=find(column_names=="Cell: Min caliper");


box_space=0.35;
% cell_lines=();
for item_idx = 1:length(lines)-1% Index for string object
    new_lines_b4_tab=lines(item_idx);
    new_lines_a4=strsplit(new_lines_b4_tab,'\t');
    new_lines_cell=cellstr(new_lines_a4);
    new_lines=cell(length(new_lines_cell),1);
    for new_line_i = 1:length(new_lines)
        check=str2double(new_lines_cell{1,new_line_i});
        if (isnan(check))
            new_lines{new_line_i,1}=new_lines_cell{1,new_line_i};
        else
            new_lines{new_line_i,1}=check;
        end
    end

    % disp(new_lines_a4_tab);

    bound_x=[new_lines{cell_centroid_x_ind}+(new_lines{cell_caliper_max})*box_space,...
              new_lines{cell_centroid_x_ind}-(new_lines{cell_caliper_max})*box_space,...
              new_lines{cell_centroid_x_ind}+(new_lines{cell_caliper_min})*box_space,...
              new_lines{cell_centroid_x_ind}-(new_lines{cell_caliper_min})*box_space...
                ];
    bound_x=[min(bound_x),max(bound_x)];

    bound_y=[new_lines{cell_centroid_y_ind}+(new_lines{cell_caliper_max})*box_space,
              new_lines{cell_centroid_y_ind}-(new_lines{cell_caliper_max})*box_space,
              new_lines{cell_centroid_y_ind}+(new_lines{cell_caliper_min})*box_space,
              new_lines{cell_centroid_y_ind}-(new_lines{cell_caliper_min})*box_space
                ];
    bound_y=[min(bound_y),max(bound_y)];

    bound_area=abs(bound_x(1)-bound_x(2))*abs(bound_y(1)-bound_y(2));

    % cell_item={new_lines{cell_centroid_x_ind},new_lines{cell_centroid_y_ind},...
    %             new_lines{cell_class},new_lines{cell_roi},new_lines{cell_area},new_lines{cell_perimeter},...
    %             new_lines{cell_caliper_max},new_lines{cell_caliper_min},bound_x,bound_y,bound_area};
    % cell_items{item_idx,1}=cell_item;

    cell_items{item_idx,1}={new_lines{cell_centroid_x_ind},new_lines{cell_centroid_y_ind},...
                new_lines{cell_class},new_lines{cell_roi},new_lines{cell_area},new_lines{cell_perimeter},...
                new_lines{cell_caliper_max},new_lines{cell_caliper_min},bound_x,bound_y,bound_area};
    listoflabels{item_idx,1}=new_lines{cell_class};
    if contains(new_lines{cell_class},'Tumor')
        class_grnd_trth(item_idx)=1;
    elseif contains(new_lines{cell_class},'Stroma')
        class_grnd_trth(item_idx)=2;
    elseif contains(new_lines{cell_class},'Immune')
        class_grnd_trth(item_idx)=3;
    else 
        class_grnd_trth(item_idx)=0;
    end


end

%% Transformation of Coordinates

bound_txed=zeros(length(lines)-1,4);

for item_idx = 1:length(lines)-1
    bound_x=cell_items{item_idx,1}{1, 9};
    bound_y=cell_items{item_idx,1}{1, 10};
    
    [r_new,c_new] = transformPointsForward(T_resize,bound_x(1),bound_y(1));
    [r_new1,c_new1] = transformPointsForward(tform_HF,r_new,c_new);
    bound_txed(item_idx,1)=r_new1;
    bound_txed(item_idx,2)=c_new1;
    
    [r_new,c_new] = transformPointsForward(T_resize,bound_x(2),bound_y(2));
    [r_new1,c_new1] = transformPointsForward(tform_HF,r_new,c_new);
    bound_txed(item_idx,3)=r_new1;
    bound_txed(item_idx,4)=c_new1;

end

%% Save Training Data
save('data_gt.mat',"bound_txed","class_grnd_trth","cell_items");