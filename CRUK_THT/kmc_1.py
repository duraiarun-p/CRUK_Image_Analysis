#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 14:10:50 2023

@author: arun
"""

import cv2
import sklearn
from sklearn import cluster
import matplotlib.pyplot as plt
import numpy as np

#%%

img=cv2.imread('/home/arun/Pictures/Test.jpg',cv2.IMREAD_UNCHANGED)
scale_percent = 35 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
# img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img0=img/255
hsv0=hsv/255
# img1=np.dstack((img0,hsv0))
img1=np.dstack((img,hsv))

#%%
# Flatten Each channel of the Image
# all_pixels  = img.reshape((-1,3))
# print(all_pixels.shape)

# #%%
# from sklearn.cluster import KMeans
# #%%


# dominant_colors = 10

# km = KMeans(n_clusters=dominant_colors)
# km.fit(all_pixels)

# centers = km.cluster_centers_

# centers = np.array(centers,dtype='uint8')



# new_img = np.zeros((dim[1],dim[0],3),dtype='uint8')

# #%%
# colors = []
# i=1
# for each_col in centers:
#     plt.subplot(1,4,i)
#     plt.axis("off")
#     i+=1
    
#     colors.append(each_col)

# #%%
cv2.imshow('Input',img)
cv2.imshow('HSV',hsv)
# # plt.figure(1)
# # plt.imshow(img)
# # plt.show()
# #%%
# for ix in range(new_img.shape[1]):
#     new_img[ix] = colors[km.labels_[ix]]
    

    
# new_img = new_img.reshape((dim[1],dim[0],3))
# plt.figure(2)
# plt.imshow(new_img)
# plt.show()
#%% OpenCV kmeans
# vectorized = img.reshape((-1,3))
vectorized = img.reshape((-1,6))
vectorized = np.float32(vectorized)

img_convert = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Set flags 
# flags = cv2.KMEANS_RANDOM_CENTERS

#image 2
K = 10
attempts=50
ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
result_image1 = res.reshape((img_convert.shape))#image 3
K = 4
# attempts=10
ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
result_image2 = res.reshape((img_convert.shape))#image 4
K = 2
# attempts=10
ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
result_image3 = res.reshape((img_convert.shape))
#%%
figure_size = 10
plt.figure(figsize=(figure_size,figure_size))#original image
plt.subplot(2,2,1),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])#image 2
plt.subplot(2,2,2),plt.imshow(result_image1)
plt.title('Segmented Image when K = 10'), plt.xticks([]), plt.yticks([])#image 3
plt.subplot(2,2,3),plt.imshow(result_image2)
plt.title('Segmented Image when K = 4'), plt.xticks([]), plt.yticks([])#image 4
plt.subplot(2,2,4),plt.imshow(result_image3)
plt.title('Segmented Image when K = 2'), plt.xticks([]), plt.yticks([])
plt.show()
#%% Sklearn kmeans

kmeans_cluster = cluster.KMeans(n_clusters=10)
kmeans_cluster.fit(vectorized)
cluster_centers = kmeans_cluster.cluster_centers_
cluster_labels = kmeans_cluster.labels_
km_mask=cluster_centers[cluster_labels].reshape(dim[1], dim[0], 3)
km_mask=km_mask.astype('uint8')
#%%
plt.figure(2,figsize=(figure_size,figure_size))
plt.imshow(km_mask)
plt.show()