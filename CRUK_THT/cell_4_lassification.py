#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 11:46:22 2023

@author: Arun PDRA, THT
"""

import os
import sys
import csv
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as pltpatch
from scipy.io import savemat,loadmat
import numpy as np
#%%
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-2_20230215/Mat_output'
base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-9_20230222/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-13_20230226/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-3_Col-5_20230218/Mat_output2'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/RT/Row-4_Col-1_20230214/Mat_output2'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/RT/Row-6_Col-10_20230223/Mat_output2'
#%% File Check

file_check='classification_QuPath.txt'

path_to_class_file=base_dir+'/'+file_check
if os.path.exists(path_to_class_file)==False:
    sys.exit(f"Execution was stopped due to cell {file_check} file was not found error")

#%% Validating bounding box by overlaying them on original figure

file_extension_type = ('.tif',) # , '.exe', 'jpg', '...')
for hist_file in os.listdir(base_dir):
    if hist_file.endswith(file_extension_type) and hist_file.startswith('R'):
        print("Found a file {}".format(hist_file)) 
        hist_img=cv2.imread(f"{base_dir}/{hist_file}")    
        
file_extension_type = ('.tiff',) # , '.exe', 'jpg', '...')
for hist_file in os.listdir(base_dir):
    if hist_file.endswith(file_extension_type) and hist_file.startswith('h'):
        print("Found a file {}".format(hist_file)) 
        hist_img_R=cv2.imread(f"{base_dir}/{hist_file}")
    

#%% Extracting information from QuPATH classification output
with open(path_to_class_file) as class_file:
    lines = [line.rstrip('\n') for line in class_file]
# using tab as a separator
items = [item.split('\t') for item in lines]

del lines

column_names=items[0] # Extracting table column names

del items[0]
# Convert numeric values into float datatype
for item_idx in range(len(items)):
    item=items[item_idx]
    for ele_idx in range(len(item)):
        ele=item[ele_idx]
        try:
            item[ele_idx]=float(ele)
        except ValueError:
            item[ele_idx]=ele
    items[item_idx]=item

#%% Extracting cell centroid from the QuPath output file

# # Index of column in items list in order

cell_centroid_x_ind=column_names.index("Centroid X px")
cell_centroid_y_ind=column_names.index("Centroid Y px")
cell_class=column_names.index("Class")
cell_roi=column_names.index("ROI")
cell_area=column_names.index("Cell: Area")
cell_perimeter=column_names.index("Cell: Perimeter")
cell_caliper_max=column_names.index("Cell: Max caliper")
cell_caliper_min=column_names.index("Cell: Min caliper")

box_space=0.35
cell_items=[]
for item_idx in range(len(items)):
    # Bounding box x co-ordinates
    bound_x=[items[item_idx][cell_centroid_x_ind]+(items[item_idx][cell_caliper_max])*box_space,
              items[item_idx][cell_centroid_x_ind]-(items[item_idx][cell_caliper_max])*box_space,
              items[item_idx][cell_centroid_x_ind]+(items[item_idx][cell_caliper_min])*box_space,
              items[item_idx][cell_centroid_x_ind]-(items[item_idx][cell_caliper_min])*box_space
                ]
    bound_x=[min(bound_x),max(bound_x)]
    
    bound_y=[items[item_idx][cell_centroid_y_ind]+(items[item_idx][cell_caliper_max])*box_space,
              items[item_idx][cell_centroid_y_ind]-(items[item_idx][cell_caliper_max])*box_space,
              items[item_idx][cell_centroid_y_ind]+(items[item_idx][cell_caliper_min])*box_space,
              items[item_idx][cell_centroid_y_ind]-(items[item_idx][cell_caliper_min])*box_space
                ]
    bound_y=[min(bound_y),max(bound_y)]
    
    bound_area=abs(bound_x[1]-bound_x[0])*abs(bound_y[1]-bound_y[0])
    
    cell_item=[items[item_idx][cell_centroid_x_ind],items[item_idx][cell_centroid_y_ind],
                items[item_idx][cell_class],items[item_idx][cell_roi],items[item_idx][cell_area],items[item_idx][cell_perimeter],
                items[item_idx][cell_caliper_max],items[item_idx][cell_caliper_min],bound_x,bound_y,bound_area]
    
    cell_items.append(cell_item)

#%% Test Points

cell_plot_index=np.arange(len(cell_items),step=1000)

fig=plt.figure(2),
plt.imshow(hist_img),
# point_ox=[]
for cell_plt_ind in range(len(cell_plot_index)):
    item_idx=cell_plot_index[cell_plt_ind]
    # print(item_idx)
    cell_item=cell_items[item_idx] 
    # point_ox.append(cell_item)
    
    plt.plot(cell_item[0],cell_item[1],marker='o', color="r")
    xy=(cell_item[-3][0],cell_item[-2][0]) # Choose min of bound x and y
    x_width=abs(cell_item[-3][0]-cell_item[-3][1])
    y_width=abs(cell_item[-2][0]-cell_item[-2][1])
    plt.plot(xy[0],xy[1],marker='o', color="k")
    rect=pltpatch.Rectangle(xy, x_width, y_width,linewidth=1, edgecolor='k', facecolor='none')
    plt.gca().add_patch(rect)