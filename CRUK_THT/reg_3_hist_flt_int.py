#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 17:56:17 2023

@author: Arun PDRA, THT
"""
import matplotlib.pyplot as plt
import cv2

#%%

flt_h=1464
flt_w=1464
px=95
py=95

cx=0.22
cy=0.22

#%%
tma_scan =  cv2.imread("/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/Row-5_Col-11_20230224/Row-5_Col-11.tif")
tma_scan=cv2.resize(tma_scan, (flt_h,flt_w))
# tma_scan[tma_scan>=255]=0
# tma_scan_blu=tma_scan[:,:,1]
# tma_scan_blu[tma_scan_blu>=235]=0

# tma_scan_f=cv2.bitwise_not(cv2.cvtColor(tma_scan,cv2.COLOR_BGR2GRAY))

tma_scan_hsv = cv2.cvtColor(tma_scan, cv2.COLOR_BGR2HSV)
tma_scan_f=tma_scan_hsv[:,:,2]
tma_scan_f[tma_scan_f>200]=0

# cv2.imshow('Output 1',tma_v)
plt.figure(1),
plt.imshow(tma_scan_f,cmap='gray')
plt.show()


# cv2.imshow('HSV image', tma_scan_f)


#%%

img1=tma_scan_f
img2=tma_scan_f

from microaligner import FeatureRegistrator, transform_img_with_tmat
freg = FeatureRegistrator()
freg.ref_img = img1
freg.mov_img = img2
transformation_matrix = freg.register()

img2_feature_reg_aligned = transform_img_with_tmat(img2, img2.shape, transformation_matrix)
