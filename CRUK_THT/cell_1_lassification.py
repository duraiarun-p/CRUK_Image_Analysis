#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 13:00:04 2023

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
base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-2_20230215/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-9_20230222/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-13_20230226/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-3_Col-5_20230218/Mat_output2'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/RT/Row-4_Col-1_20230214/Mat_output2'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/RT/Row-6_Col-10_20230223/Mat_output2'
#%% File Check

file_check='classification_Qupath.txt'

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
    
    # fig=plt.figure(2),
    # plt.plot(cell_item[0],cell_item[1],marker='o', color="r")
    # xy=(cell_item[-3][0],cell_item[-2][0]) # Choose min of bound x and y
    # x_width=abs(cell_item[-3][0]-cell_item[-3][1])
    # y_width=abs(cell_item[-2][0]-cell_item[-2][1])
    # plt.plot(xy[0],xy[1],marker='o', color="k")
    # rect=pltpatch.Rectangle(xy, x_width, y_width,linewidth=1, edgecolor='k', facecolor='none')
    # plt.gca().add_patch(rect)
    # plt.imshow(hist_img)
    # plt.show()
    
    cell_items.append(cell_item)

        

#%% 

# fig=plt.figure(1),
# plt.plot(cell_item[0],cell_item[1],marker='o', color="r")
# xy=(cell_item[-3][0],cell_item[-2][0]) # Choose min of bound x and y
# x_width=abs(cell_item[-3][0]-cell_item[-3][1])
# y_width=abs(cell_item[-2][0]-cell_item[-2][1])
# plt.plot(xy[0],xy[1],marker='o', color="k")
# rect=pltpatch.Rectangle(xy, x_width, y_width,linewidth=1, edgecolor='k', facecolor='none')
# plt.gca().add_patch(rect)
# plt.imshow(hist_img)

# del cell_item

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
    
#%%


tx_siz=hist_img_R.shape
ox_siz=hist_img.shape
tx_f_siz=np.round((np.array(ox_siz)/np.array(tx_siz)))
tx_f_siz1=((np.array(tx_siz)/np.array(ox_siz)))# Proper scale factor for this implemetnation

hist_img_R1=cv2.resize(hist_img,(tx_siz[0],tx_siz[1]), interpolation= cv2.INTER_NEAREST)


# cell_item_tx_f[0]=(cell_item_tx[0]/ox_siz[0])*tx_siz[0]
# cell_item_tx_f[1]=(cell_item_tx[1]/ox_siz[1])*tx_siz[1]

# ox_mid=np.round(np.array(ox_siz[:2])/2-1)
# tx_mid=np.round(np.array(tx_siz[:2])/2-1)
ox_mid=(np.array(ox_siz[:2])-1/2)
tx_mid=(np.array(tx_siz[:2])-1/2)
# ox_mid=np.round(ox_mid)
# tx_mid=np.round(tx_mid)

tx_mid1=np.zeros_like(ox_mid)
# tx_mid1[0]=(ox_mid[0]/ox_siz[0])*tx_siz[0]
# tx_mid1[1]=(ox_mid[1]/ox_siz[1])*tx_siz[1]
# tx_mid1=np.round(tx_mid1)
tx_mid1[0]=(ox_mid[0]-ox_mid[0])*tx_f_siz1[0]+tx_mid[0]
tx_mid1[1]=(ox_mid[1]-ox_mid[1])*tx_f_siz1[1]+tx_mid[1]


# ScalingFactor=np.array(tx_siz)/np.array(ox_siz)

# T1 = TranslateFactor + ScalingFactor*S1
# T2 = TranslateFactor + ScalingFactor*S2
# TranslateFactor=np.zeros_like(ScalingFactor)
# TranslateFactor=tx_mid-ox_mid





#  pixel coordinates resize
#  pixel coordinates transform
#  do it for one cell item
#  do it for all items through loop

cell_item_tx=np.array(cell_items[0][:2])
# cell_item_tx_f=(ScalingFactor[:2]*cell_item_tx)-TranslateFactor[:2]
# cell_item_tx_f=(ScalingFactor[:2]*cell_item_tx)
cell_item_tx_f=np.zeros_like(cell_item_tx)
# cell_item_tx_f=(cell_item_tx*tx_mid[:2])/ox_mid[:2]

# cell_item_tx_f[0] = (cell_item_tx[0] - ox_mid[0])*ScalingFactor[0] + tx_mid[0]
# cell_item_tx_f[1] = (cell_item_tx[1] - ox_mid[1])*ScalingFactor[1] + tx_mid[1]



# TranslateFactor = (tx_mid[1] *ox_mid[0] - tx_mid[0] *ox_mid[1]) / (ox_mid[0] - ox_mid[1])
# ScalingFactor   = (tx_mid[1] - tx_mid[0]) / (ox_mid[1] - ox_mid[0])


# cell_item_tx_f[0] = (cell_item_tx[0] - ox_mid[0])*ScalingFactor[0] + tx_mid[0]
# cell_item_tx_f[1] = (cell_item_tx[1] - ox_mid[1])*ScalingFactor[1] + tx_mid[1]

# cell_item_tx_f[0] = (cell_item_tx[0] - ox_mid[0])*ScalingFactor[0] + tx_mid[0]
# cell_item_tx_f[1] = (cell_item_tx[1] - ox_mid[1])*ScalingFactor[1] + tx_mid[1]

# cell_item_tx_f[0] = TranslateFactor + ScalingFactor*cell_item_tx[0]
# cell_item_tx_f[1] = TranslateFactor + ScalingFactor*cell_item_tx[1]

# cell_item_tx_f[0] = ScalingFactor*cell_item_tx[0]
# cell_item_tx_f[1] = ScalingFactor*cell_item_tx[1]

# cell_item_tx_f[0]=cell_item_tx[0]*tx_f_siz[0]
# cell_item_tx_f[1]=cell_item_tx[1]*tx_f_siz[1]

# cell_item_tx_f[0]=((cell_item_tx[0]-ox_mid[0])*tx_f_siz[0]) + tx_mid[0]
# cell_item_tx_f[1]=((cell_item_tx[1]-ox_mid[1])*tx_f_siz[1]) + tx_mid[1]

# cell_item_tx_f[0]=((ox_mid[0]-cell_item_tx[0])/tx_f_siz[0]) + tx_mid[0]
# cell_item_tx_f[1]=((ox_mid[1]-cell_item_tx[1])/tx_f_siz[1]) + tx_mid[1]


#%%
fig=plt.figure(3),
plt.imshow(hist_img_R1),
point_ox=[]
point_tx=[]
# plt.plot(cell_item_tx_f[0],cell_item_tx_f[1],marker='o', color="r")
for cell_plt_ind in range(len(cell_plot_index)):
    item_idx=cell_plot_index[cell_plt_ind]
    cell_item_tx=np.array(cell_items[item_idx][:2])
    cell_item_tx_f=np.zeros_like(cell_item_tx)
    cell_item_tx_f[0]=((cell_item_tx[0]-ox_mid[0])*tx_f_siz1[0]) + tx_mid[0]
    cell_item_tx_f[1]=((cell_item_tx[1]-ox_mid[1])*tx_f_siz1[1]) + tx_mid[1]
    point_ox.append(cell_item_tx)
    point_tx.append(cell_item_tx_f)
#     cell_item_tx_f[0]=((cell_item_tx[0]-ox_mid[0])/tx_f_siz[0]) + tx_mid[0]
#     cell_item_tx_f[1]=((cell_item_tx[1]-ox_mid[1])/tx_f_siz[1]) + tx_mid[1]
#     plt.plot(cell_item_tx_f[0],cell_item_tx_f[1],marker='o', color="k")
#     cell_item_tx_f[0]=cell_item_tx[0]/tx_f_siz[0]
#     cell_item_tx_f[1]=cell_item_tx[1]/tx_f_siz[1]
#     plt.plot(cell_item_tx_f[0],cell_item_tx_f[1],marker='o', color="r")
#     cell_item_tx_f[0]=np.round(cell_item_tx[0]/tx_f_siz[0])
#     cell_item_tx_f[1]=np.round(cell_item_tx[1]/tx_f_siz[1])
#     plt.plot(cell_item_tx_f[0],cell_item_tx_f[1],marker='o', color="b")
#     cell_item_tx_f[0]=np.round(cell_item_tx[0]*tx_f_siz1[0])
#     cell_item_tx_f[1]=np.round(cell_item_tx[1]*tx_f_siz1[1])
    plt.plot(cell_item_tx_f[0],cell_item_tx_f[1],marker='o', color="g")
#%%
fig=plt.figure(20),
plt.imshow(hist_img),
plt.plot(ox_mid[0],ox_mid[1],marker='o', color="r")

fig=plt.figure(30),
plt.imshow(hist_img_R1),
plt.plot(tx_mid[0],tx_mid[1],marker='o', color="r")
plt.plot(tx_mid1[0],tx_mid1[1],marker='o', color="g")
#%%
mdic={'point_ox':point_ox,'point_tx':point_tx}
savemat('points.mat',mdic)