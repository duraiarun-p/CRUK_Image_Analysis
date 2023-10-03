#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:17:16 2023

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

file_check='classification_QuPath.txt'

path_to_class_file=base_dir+'/'+file_check
if os.path.exists(path_to_class_file)==False:
    sys.exit(f"Execution was stopped due to cell {file_check} file was not found error")

#%% Loading all files required for coordinate transform

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
        
file_extension_type = ('.mat',) # , '.exe', 'jpg', '...')
for hist_file in os.listdir(base_dir):
    if hist_file.endswith(file_extension_type) and hist_file.startswith('w'):
        print("Found a file {}".format(hist_file)) 
        wm_mat_contents=loadmat(f"{base_dir}/{hist_file}")
        # core_mat_contents_list=list(core_mat_contents.keys())
        wm_mat_contents_ref=wm_mat_contents['warp_matrix']
        warp_matrix=wm_mat_contents_ref[()]
        
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

siz_tx=hist_img_R.shape # Registered Image Shape
siz_ox=hist_img.shape # Original Image Shape



cell_plot_index=np.arange(len(cell_items),step=1000)

fig=plt.figure(2),
plt.imshow(hist_img),
plt.plot(siz_ox[0]/2,siz_ox[1]/2,marker='o', color="g")
point_ox=[]
for cell_plt_ind in range(len(cell_plot_index)):
    item_idx=cell_plot_index[cell_plt_ind]
    # print(item_idx)
    cell_item=cell_items[item_idx] 
    cell_item_ox=cell_item[:2]
    # cell_item_ox.append(1)
    point_ox.append(cell_item_ox)
    
    plt.plot(cell_item[0],cell_item[1],marker='o', color="r")
    xy=(cell_item[-3][0],cell_item[-2][0]) # Choose min of bound x and y
    x_width=abs(cell_item[-3][0]-cell_item[-3][1])
    y_width=abs(cell_item[-2][0]-cell_item[-2][1])
    plt.plot(xy[0],xy[1],marker='o', color="k")
    rect=pltpatch.Rectangle(xy, x_width, y_width,linewidth=1, edgecolor='k', facecolor='none')
    plt.gca().add_patch(rect)
point_ox=np.array(point_ox)

# fig=plt.figure(3),
# plt.imshow(hist_img_R)
#%%
warp_matrix_inv_1=np.linalg.pinv(warp_matrix)
warp_matrix_tp=np.transpose(warp_matrix)

warp_matrix_inv_2=np.zeros_like(warp_matrix)
for i in range(2):
    for j in range(2):
        warp_matrix_inv_2[i,j]=1/(warp_matrix[i,j])

# warp_matrix[:2,:2]=warp_matrix_inv_2[:2,:2]

# warp_matrix=warp_matrix/2

mid_ox=np.array(siz_ox[:2])/2-1# Mid point of original image

mid_tx=np.array(siz_tx[:2])/2-1# Mid point of registered image

tx_f_siz1=((np.array(siz_tx)/np.array(siz_ox)))# 
#%% Points were rescaled to new size
hist_img_R1=cv2.resize(hist_img,(siz_tx[1],siz_tx[0]), interpolation= cv2.INTER_NEAREST)

points_ox=[]
plt.figure(3)
plt.imshow(hist_img_R1)
for cell_plt_ind in range(len(cell_plot_index)):
    item_idx=cell_plot_index[cell_plt_ind]
    cell_item_tx=cell_items[item_idx][:2]
    cell_item_tx.append(1)
    cell_item_tx=np.array(cell_item_tx,dtype=np.int32)
    # c
    cell_item_tx_f=np.zeros_like(cell_item_tx)
    cell_item_tx_f[0]=((cell_item_tx[0]-mid_ox[0])*tx_f_siz1[0]) + mid_tx[0]
    cell_item_tx_f[1]=((cell_item_tx[1]-mid_ox[1])*tx_f_siz1[1]) + mid_tx[1]
    cell_item_tx_f[2]=1
    # np.append(cell_item_tx_f,1)
    # old_Pnt=cell_item_tx-[siz_ox[1],siz_ox[0],1]
    # cell_item_tx_f=cell_item_tx
    old_Pnt=cell_item_tx_f
    # old_Pnt=cell_item_tx_f/[siz_tx[0],siz_tx[1],1]
    points_ox.append(old_Pnt)
    
    
    
    # old_Pnt=cell_item_tx/[siz_ox[0],siz_ox[1],1]
    # old_Pnt=old_Pnt+[mid_tx[0],mid_tx[1],0]
# #     # nP1=homography@oP1
    # new_Pnt=np.dot(old_Pnt,warp_matrix)
    
    # new_Pnt=new_Pnt*[siz_tx[0],siz_tx[1]]
    
#     # new_Pnt=cv2.warpAffine(cell_item_tx,warp_matrix,(siz_tx[1],siz_tx[0]))
#     # new_Pnt=cv2.warpPerspective(cell_item_tx, warp_matrix, (siz_tx[1], siz_tx[0]))
    # new_Pnt=np.dot(old_Pnt,warp_matrix_inv_1)
    # new_Pnt=new_Pnt+[mid_ox[0],mid_ox[1]]
    # new_Pnt=new_Pnt*[siz_tx[0],siz_tx[1]]
    # new_Pnt[0]=new_Pnt[0]-warp_matrix_inv[-1][0]
    # new_Pnt[1]=new_Pnt[1]-warp_matrix_inv[-1][1]
    # new_Pnt=cv2.transform(cell_item_tx,warp_matrix)
    
    # new_Pnt=np.dot(old_Pnt,warp_matrix_tp)
    # new_Pnt=np.matmul(old_Pnt.reshape(-1),warp_matrix).reshape(3,-1)
    # new_Pnt=new_Pnt*[siz_tx[0],siz_tx[1],1]
    # new_Pnt=new_Pnt+[mid_ox[0],mid_ox[1]]
    # new_Pnt[0]=new_Pnt[0]-siz_tx[0]
    # new_Pnt[1]=new_Pnt[1]-siz_tx[1]
    # print(new_Pnt)
    # plt.plot(new_Pnt[0],new_Pnt[1],marker='o', color="r")
    plt.plot(old_Pnt[0],old_Pnt[1],marker='x', color="r")

    # old_Pnt=
#%% Apply transformation
# all points were rescaled to the reg height and width
# points_ox=points_ox/[siz_tx[0],siz_tx[1],1]
points_ox=np.array(points_ox)
points_ox=np.transpose(points_ox)
points_ox1=points_ox
points_tx=np.zeros_like(points_ox)
# points_ox1=points_ox/[siz_tx[0],siz_tx[1],1]
# points_ox1=np.divide(points_ox1,[siz_tx[0],siz_tx[1],1])
# for ci in range(len(cell_plot_index)):
    # points_ox1_t=points_ox1[:,ci]
#     points_ox1_t=points_ox1_t/[siz_tx[0],siz_tx[1],1]
    # points_ox1[:,ci]=points_ox1[:,ci]/[siz_tx[0],siz_tx[1],1]
    
# points_ox1=points_ox*np.transpose([tx_f_siz1[0],tx_f_siz1[1],1])
# points_tx=np.matmul(warp_matrix,points_ox[0])
# warp_matrix_tp[-1,:]=2*(warp_matrix_tp[-1,:])

# points_tx1=points_ox1@warp_matrix # This is where the trouble is
# warp_matrix=np.flipud(warp_matrix)
points_tx1=warp_matrix@points_ox1
points_tx=points_tx1

mid_tx_1=np.hstack((mid_tx,1))
mid_tx_f=warp_matrix@mid_tx_1

dis_c=mid_tx_f-mid_tx
# mid_tx_f=mid_tx_f-dis_c
# points_translation=np.flipud(warp_matrix[:,-1])
# points_translation=np.hstack((points_translation,0))
# points_tx=points_tx1*[siz_tx[0],siz_tx[1]]
# points_tx=points_tx-warp_matrix[:,-1]
for points_idx in range(len(cell_plot_index)):
    # points=points_ox[points_idx][0:2]
    points=points_tx[:,points_idx]
#     points_translation=points_tx[points_idx][-1]
    # points=points-points_translation
#     # points=points+abs(warp_matrix[:,-1])
    # points[0]=((points[0]-mid_tx[0])*tx_f_siz1[0]) + mid_tx[0]
    # points[1]=((points[1]-mid_tx[1])*tx_f_siz1[1]) + mid_tx[1]
    # points[0]=((points[0]-mid_tx[0])*tx_f_siz1[0])
    # points[1]=((points[1]-mid_tx[1])*tx_f_siz1[1])
    # points[0]=((points[0]+mid_tx[0]))
    # points[1]=((points[1]+mid_tx[1]))
    # points[0]=((points[0]-mid_tx[0]))
    # points[1]=((points[1]-mid_tx[1]))
    # points[0]=((points[0]-dis_c[0]))
    # points[1]=((points[1]-dis_c[1]))
    # points[0]=((points[0]/tx_f_siz1[0]))
    # points[1]=((points[1]/tx_f_siz1[1]))
    # points[0]=((points[0]*siz_tx[0]))
    # points[1]=((points[1]*siz_tx[1]))
    
    # oP1=points
    # nP1=oP1[1]*warp_matrix[0,0]+oP1[1]*warp_matrix[0,1]+warp_matrix[0,2]
    # nP2=oP1[0]*warp_matrix[1,0]+oP1[0]*warp_matrix[1,1]+warp_matrix[1,2]
    # points_tx[1,points_idx]=nP1
    # points_tx[0,points_idx]=nP2
    # points1=np.array([])
    # points_tx[points_idx][0:2]=points
    points_tx[:,points_idx]=points[:2]
#     print(points)



oP1=points_ox[:,0]
# oP1=np.transpose(oP1)
nP1=oP1[0]*warp_matrix[0,0]+oP1[0]*warp_matrix[0,1]+warp_matrix[0,2]
nP2=oP1[1]*warp_matrix[1,0]+oP1[1]*warp_matrix[1,1]+warp_matrix[1,2]


A, t = np.hsplit(warp_matrix[1:].T/(-warp_matrix[0])[:,None], [2])
t = np.transpose(t)[0]

# nP=cv2.transform(oP1,warp_matrix)[0,:,:]
# nP=cv2.perspectivetransform(oP1,warp_matrix)
oP1 = np.expand_dims(oP1, axis=1)
# oP1=np.transpose(oP1)
nP = cv2.transform(oP1, warp_matrix_tp, oP1.shape)[:,0]
nP = np.squeeze(nP[:,0])

# p = (50,100) # your original point
# px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
# py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
# p_after = (int(px), int(py)) # after transformation

# points_tx=np.transpose(points_tx)
# points_tx=points_tx/[siz_ox[0],siz_ox[1]]
# points_tx=points_tx*[siz_tx[0],siz_tx[1]]
# points_tx=np.transpose(points_tx)
#%%
# item_idx=cell_plot_index[cell_plt_ind]
# print(item_idx)
# old_Pnts=cell_items[cell_plot_index] 
# point_tx=cv2.transform(point_ox,warp_matrix)
# points_ox=list(points_ox)
# points_tx=list(points_tx)
plt.figure(4)
plt.imshow(hist_img_R)
plt.plot(mid_tx_f[0],mid_tx_f[1],marker='o', color="g")
plt.plot(nP1,nP2,marker='o', color="r")
plt.plot(nP[0],nP[1],marker='o', color="c")
# for p1_idx in range(len(cell_plot_index)):
#     # p1=points_ox[p1_idx]
#     p1=points_ox[:,p1_idx]
#     plt.plot(p1[0],p1[1],marker='o', color="b")  

# for p2_idx in range(len(cell_plot_index)):
#     # p2=points_tx[p2_idx]
#     p2=points_tx[:,p2_idx]
#     plt.plot(p2[0],p2[1],marker='o', color="r")  
# # for p1,p2,p3 in points_tx:
# #     plt.plot(p1,p2,marker='o', color="b")
# plt.plot(mid_tx[0],mid_tx[1],marker='o', color="k")