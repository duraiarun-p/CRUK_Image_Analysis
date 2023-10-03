#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:37:10 2023

@author: Arun PDRA, THT
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import hausdorff_distance as hd
from skimage.metrics import normalized_mutual_information as nmi
from skimage.metrics import normalized_root_mse as nrmse

#%%

flt_h=1464
flt_w=1464
px=95
py=95

cx=0.22
cy=0.22

#%%
tma_scan =  cv2.imread("/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/Row-5_Col-11_20230224/Row-5_Col-11.tif")
tma_int_fiji=cv2.imread('/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/CRUK_THT/Test_Output/Tiling/Int_Core.png')
tma_int_sbm=cv2.imread('/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/CRUK_THT/Test_Output/Tiling/Core_Int_SBM.png')

# /home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/CRUK_THT/Test_Output/Tiling/Row-1_Col-9_20230222
# mypath='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/Row-4_Col-7_20230220/FLT_IMG_DIR'

# tma_scan =  cv2.imread("/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/Row-1_Col-9_20230222/Row-1_Col-9_1.tif")
# tma_int_fiji=cv2.imread('/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/CRUK_THT/Test_Output/Tiling/Row-1_Col-9_20230222/Int_Core.tif')
# tma_int_sbm=cv2.imread('/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/CRUK_THT/Test_Output/Tiling/Row-1_Col-9_20230222/Core_Int_SBM_1.png')

# tma_scan =  cv2.imread("/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/Row-4_Col-7_20230220/Row-4_Col-7.tif")
# tma_int_fiji=cv2.imread('/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/CRUK_THT/Test_Output/Tiling/Row-4_Col-7_20230220/Int_Core.png')
# tma_int_sbm=cv2.imread('/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/CRUK_THT/Test_Output/Tiling/Row-4_Col-7_20230220/Core_Int_SBM_2.png')

#%%


tma_scan=cv2.resize(tma_scan, (flt_h,flt_w))
plt.figure(100),
plt.imshow(tma_scan,cmap='gray')
plt.show()
plt.savefig('HE.png')
# tma_scan[tma_scan>=255]=0
# tma_scan_blu=tma_scan[:,:,1]
# tma_scan_blu[tma_scan_blu>=235]=0

# tma_scan_f=cv2.bitwise_not(cv2.cvtColor(tma_scan,cv2.COLOR_BGR2GRAY))

tma_scan_hsv = cv2.cvtColor(tma_scan, cv2.COLOR_BGR2HSV)
tma_scan_f=tma_scan_hsv[:,:,2]
tma_scan_f[tma_scan_f>200]=0
tma_scan_f[tma_scan_f<190]=0

# cv2.imshow('Output 1',tma_v)
plt.figure(1),
plt.imshow(tma_scan_f,cmap='gray')
plt.show()
#%%



tma_int_fiji = cv2.rotate(tma_int_fiji, cv2.ROTATE_90_COUNTERCLOCKWISE)

tma_int_fiji=np.fliplr(tma_int_fiji)

# tma_int_fiji=np.squeeze(tma_int_fiji,axis=2)
tma_int_fiji=tma_int_fiji[:,:,0]
tma_int_fiji=cv2.resize(tma_int_fiji, (flt_h,flt_w))



plt.figure(2),
plt.imshow(tma_int_fiji,cmap='gray')
plt.show()
#%%


# tma_int_sbm = cv2.rotate(tma_int_sbm, cv2.ROTATE_90_COUNTERCLOCKWISE)

# tma_int_sbm=np.fliplr(tma_int_sbm)
tma_int_sbm=tma_int_sbm[:,:,0]


plt.figure(3),
plt.imshow(tma_int_sbm,cmap='gray')
plt.show()
#%% Stitching Validation using Image Registration

def Affine_OpCV_2D(Fixed_sitk,Moving_sitk):
    sz=Fixed_sitk.shape
    warp_mode = cv2.MOTION_AFFINE

    warp_matrix = np.eye(2, 3, dtype=np.float32)

    number_of_iterations = 50000


    termination_eps = 1e-10

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    (cc, warp_matrix) = cv2.findTransformECC (Fixed_sitk,Moving_sitk,warp_matrix, warp_mode, criteria)

    # # warp_matrix=cv2.getAffine

    Moving_sitk_registered = cv2.warpAffine(Moving_sitk, warp_matrix, (sz[0],sz[1]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    
    return Moving_sitk_registered, warp_matrix, cc

def Warp_via_Matrix_OpCV(Fixed_sitk,Moving_sitk,warp_matrix):
    sz=Fixed_sitk.shape
    Moving_sitk_registered = cv2.warpAffine(Moving_sitk, warp_matrix, (sz[0],sz[1]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return Moving_sitk_registered

def Warp_via_Matrix_OpCV_2D_3D(Fixed_sitk,Moving_sitk,warp_matrix):
    sz=Fixed_sitk.shape
    Moving_sitk_registered=np.zeros((sz[0],sz[1],3))
    for page in range(3):
        Moving_sitk_registered[:,:,page] = cv2.warpAffine(Moving_sitk[:,:,page], warp_matrix, (sz[0],sz[1]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return Moving_sitk_registered
#%%
Fixed=tma_scan_f
Moving=tma_int_fiji
# Moving=tma_scan_f
# Fixed=tma_int_fiji

Moving_R, warp_matrix, cc=Affine_OpCV_2D(Fixed,Moving)
Moving_R=Moving_R.astype('uint8')
# DIff=np.abs(Moving_R-Moving)
# RE=np.sum(DIff)
(SSIM,DIff1)=ssim(Fixed, Moving_R, full=True)
NRMSE=nrmse(Fixed,Moving_R)
NMI=nmi(Fixed,Moving_R)
# Fixed_mask, thresh1 = cv2.threshold(Fixed, 10, 255, cv2.THRESH_BINARY)
# Moving_R_mask, thresh1 = cv2.threshold(Moving_R, 10, 255, cv2.THRESH_BINARY)
HD=hd(Fixed,Moving_R)
# HD=hd(Fixed_mask,Moving_R_mask)
Perf_score_FH=np.array([SSIM,NRMSE,NMI,HD])
# Perf_score_FH=[SSIM,NRMSE,NMI,HD]
print(Perf_score_FH)

plt.figure(4),
# plt.subplot(2,2,1)
# plt.imshow(Moving,cmap='gray')
# plt.subplot(2,2,2)
# plt.imshow(Fixed,cmap='gray')
# plt.subplot(2,2,3)
plt.imshow(Moving_R,cmap='gray')
# plt.subplot(2,2,4)
# plt.imshow(DIff,cmap='gray')
plt.title('Fiji')
plt.show()
plt.savefig('Mosaic_FIji.png')

Fixed=tma_scan_f
Moving=tma_int_sbm

# Moving=tma_scan_f
# Fixed=tma_int_sbm

Moving_R2, warp_matrix, cc=Affine_OpCV_2D(Fixed,Moving)
Moving_R2=Moving_R2.astype('uint8')
# DIff=np.abs(Moving_R-Moving)
# RE=np.sum(DIff)
# (RE,DIff1)=ssim(Fixed, Moving_R, full=True)
(SSIM,DIff1)=ssim(Fixed, Moving_R2, full=True)
NRMSE=nrmse(Fixed,Moving_R2)
NMI=nmi(Fixed,Moving_R2)
# Fixed_mask, thresh1 = cv2.threshold(Fixed, 10, 255, cv2.THRESH_BINARY)
# Moving_R_mask, thresh1 = cv2.threshold(Moving_R, 10, 255, cv2.THRESH_BINARY)
HD=hd(Fixed,Moving_R2)
# HD=hd(Fixed_mask,Moving_R_mask)
Perf_score_FS=np.array([SSIM,NRMSE,NMI,HD])
# Perf_score_FS=[SSIM,NRMSE,NMI,HD]
print(Perf_score_FS)

# Perf_dev=((Perf_score_FH-Perf_score_FS) /((Perf_score_FH+Perf_score_FS)*0.5))*100
# print(Perf_dev)

Dev=np.abs(Perf_score_FH-Perf_score_FS)
Avg=(Perf_score_FH+Perf_score_FS)*0.5

Avg1=np.array([1,Avg[1],1,Avg[3]])
Dev_per=(Dev/Avg1)*100

print(Dev_per)

plt.figure(5),
# plt.subplot(2,2,1)
# plt.imshow(Moving,cmap='gray')
# plt.subplot(2,2,2)
# plt.imshow(Fixed,cmap='gray')
# plt.subplot(2,2,3)
plt.imshow(Moving_R2,cmap='gray')
# plt.subplot(2,2,4)
# plt.imshow(DIff,cmap='gray')
plt.title('SBM')
plt.show()
plt.savefig('Mosaaic_SBM.png')

#%%
# tma_scan_R=Warp_via_Matrix_OpCV(Fixed,tma_scan[:,:,1],warp_matrix)
# tma_scan_R=Warp_via_Matrix_OpCV_2D_3D(Fixed,tma_scan,warp_matrix)
#%%
# cv2.imshow('Hist',tma_scan)
# cv2.imshow('Registered Hist',tma_scan_R)
# plt.figure(6),
# plt.imshow(tma_scan)
# plt.show()

# plt.figure(7),
# plt.imshow(tma_scan_R)
# plt.show()