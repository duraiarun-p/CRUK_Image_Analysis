#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 18:20:38 2023

@author: Arun PDRA, THT
"""
#%%
import os
import sys
import cv2
import numpy as np
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import normalized_mutual_information as nmi
from skimage.metrics import structural_similarity as ssim

from scipy.io import savemat,loadmat
import time
#%% Masking function for the Histology Image

def coreg_img_pre_process(hist_img,thresh):
    hist_img_gray=cv2.cvtColor(hist_img, cv2.COLOR_BGR2GRAY)
    hist_img_gray_f=hist_img_gray
    hist_img_gray[hist_img_gray>thresh]=0

    hist_img_hsv = cv2.cvtColor(hist_img, cv2.COLOR_BGR2HSV)
    hist_img_val=hist_img_hsv[:,:,2]
    hist_img_val[hist_img_val>thresh]=0

    hist_img_int=cv2.bitwise_or(hist_img_gray,hist_img_val)
    hist_img_int[hist_img_int>thresh]=0
    ##%% Mask for circular ROI
    hist_img_msk=hist_img_int
    hist_img_msk[hist_img_msk>0]=255

    # hist_img_msk_inv=cv2.bitwise_not(hist_img_msk)

    h, w = hist_img_int.shape[:2]
    # mask = np.zeros((h+2, w+2), np.uint8)

    hist_img_msk_blur = cv2.GaussianBlur(hist_img_msk, (35,35),100)
    hist_img_msk_edge = cv2.Sobel(src=hist_img_msk_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=25)

    hist_img_msk_edge_con=hist_img_msk_edge
    hist_img_msk_edge_con[hist_img_msk_edge_con==0]=255
    hist_img_msk_edge_con[hist_img_msk_edge_con!=255]=0
    hist_img_msk_edge_con=hist_img_msk_edge_con.astype('uint8')

    hist_img_msk_edge_con_edge=cv2.Sobel(src=hist_img_msk_edge_con, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=25)
    hist_img_msk_edge_con_edge[hist_img_msk_edge_con_edge==0]=255
    hist_img_msk_edge_con_edge[hist_img_msk_edge_con_edge!=255]=0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(35,35))
    hist_img_msk_edge_con_edge = cv2.morphologyEx(hist_img_msk_edge_con_edge,cv2.MORPH_OPEN,kernel)
    hist_img_msk_edge_con_inv=cv2.bitwise_not(hist_img_msk_edge_con)


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))
    hist_img_msk_edge_con_inv = cv2.morphologyEx(hist_img_msk_edge_con_inv,cv2.MORPH_OPEN,kernel)

    contour,hier = cv2.findContours(hist_img_msk_edge_con_inv,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contour:
        cv2.drawContours(hist_img_msk_edge_con_inv,[cnt],0,255,-1)

    gray = cv2.bitwise_not(hist_img_msk_edge_con_inv)

    hist_mask = cv2.bitwise_not(gray)# Core mask

    hist_img_gray_f=cv2.bitwise_and(hist_img_gray_f,hist_mask)# Core mask applied
    
    hist_mask_1=np.array([hist_mask,hist_mask,hist_mask])
    hist_mask_1=np.moveaxis(hist_mask_1,0,2)
    
    hist_img_f=cv2.bitwise_and(hist_img,hist_mask_1)
    hist_img_hsv_f=cv2.bitwise_and(hist_img_hsv,hist_mask_1)

    return hist_img_hsv_f,hist_img_f,hist_img_gray_f,hist_mask,hist_img_gray




#%% Registration by Homography

# imRef_grey - Fixed Image, imgTest_grey - Moving Image
#Prepare data

def my_norm(x):
    x_norm = (x-np.min(x))/(np.max(x)-np.min(x))
    return x_norm


def prepare_img_4_reg_Fixed_changedatatype(Fixed,Moving):
    Fixed_datatype=str(Fixed.dtype)
    Moving_datatype=str(Moving.dtype)
    
    Fixed_N=np.zeros_like(Fixed)
    
    # Fixed=np.ascontiguousarray(Fixed,dtype=Moving_datatype)
    # Fixed_N=np.ascontiguousarray(Fixed_N,dtype=Moving_datatype)

    # Fixed_N = np.round(cv2.normalize(Fixed,  Fixed_N, 0, 255, cv2.NORM_MINMAX))
    # Fixed_N = cv2.normalize(Fixed,  Fixed_N, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    Fixed_N = my_norm(Fixed)*255 # for FLT intensity range to [0 255] image intensity range
    Fixed_N = Fixed_N.astype(Moving_datatype)
    
    Moving_N=np.zeros_like(Moving)
    Moving_N = np.round(cv2.normalize(Moving,  Moving_N, 0, 255, cv2.NORM_MINMAX))
    return Fixed_N, Moving_N


def prepare_img_4_reg_Moving_changedatatype(Fixed,Moving):
    Fixed_datatype=str(Fixed.dtype)
    Moving_datatype=str(Moving.dtype)
    
    Fixed_N=np.zeros_like(Fixed)
    # Fixed=np.ascontiguousarray(Fixed,dtype=Fixed_datatype)
    # Fixed_N=np.ascontiguousarray(Fixed_N,dtype=Fixed_datatype)
    
    # Fixed_N = np.round(cv2.normalize(Fixed,  Fixed_N, 0, 255, cv2.NORM_MINMAX))
    Fixed_N = my_norm(Fixed)*255
    Fixed_N = Fixed_N.astype(Fixed_datatype)
    
    Moving_N=np.zeros_like(Moving)
    Moving = Moving.astype(Fixed_datatype)
    Moving_N = Moving_N.astype(Fixed_datatype)
    
    Moving_N = np.round(cv2.normalize(Moving,  Moving_N, 0, 255, cv2.NORM_MINMAX))
    
    return Fixed_N, Moving_N
    

#OpenCV - Homography with 2D image as inputs
def OCV_Homography_2D(imgRef_grey,imgTest_grey,NFN):
    height, width = imgRef_grey.shape
    imgTest_grey=cv2.resize(imgTest_grey, (width,height), interpolation= cv2.INTER_NEAREST)
    orb_detector = cv2.ORB_create(nfeatures=NFN)
 
    # Extract key points and descriptors for both images
    keyPoint1, des1 = orb_detector.detectAndCompute(imgTest_grey, None)
    keyPoint2, des2 = orb_detector.detectAndCompute(imgRef_grey, None)
    
    # # Initiate SIFT detector
    # sift_detector = cv2.SIFT_create()
    # # Find the keypoints and descriptors with SIFT
    # keyPoint1, des1 = sift_detector.detectAndCompute(imgTest_grey, None)
    # keyPoint2, des2 = sift_detector.detectAndCompute(imgRef_grey, None)
    
    
    # Match features between two images using Brute Force matcher with Hamming distance
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matcher = cv2.BFMatcher()
    # Match the two sets of descriptors.
    matches = matcher.match(des1, des2)
     
    # # Sort matches on the basis of their Hamming distance.
    matches = sorted(matches, key=lambda x: x.distance)
    
     
    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches) * 0.9)]
    
    # matches = matcher.knnMatch(des1, des2, k=2)

    # # Filter out poor matches
    # good_matches = []
    # for m,n in matches:
    #     if m.distance < 0.9*n.distance:
    #         good_matches.append(m)
    
    # matches = good_matches
    
    
    no_of_matches = len(matches)
    
        # Define 2x2 empty matrices
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))
     
    # Storing values to the matrices
    for i in range(len(matches)):
        p1[i, :] = keyPoint1[matches[i].queryIdx].pt
        p2[i, :] = keyPoint2[matches[i].trainIdx].pt
     
    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    
    # Use homography matrix to transform the unaligned image wrt the reference image.
    aligned_img = cv2.warpPerspective(imgTest_grey, homography, (width, height))
    return aligned_img, homography, mask

# Affine OpenCV with 2D image as inputs
def Affine_OpCV_2D(Fixed_sitk,Moving_sitk,number_of_iterations):
    sz=Fixed_sitk.shape
    Moving_sitk=cv2.resize(Moving_sitk, (sz[1],sz[0]), interpolation= cv2.INTER_NEAREST)
    
    warp_mode = cv2.MOTION_AFFINE

    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # number_of_iterations = 10000


    termination_eps = 1e-10

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    (cc, warp_matrix) = cv2.findTransformECC (Fixed_sitk,Moving_sitk,warp_matrix, warp_mode, criteria)

    # # warp_matrix=cv2.getAffine

    Moving_sitk_registered = cv2.warpAffine(Moving_sitk, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    
    return Moving_sitk_registered, warp_matrix, cc
#%% Performation
def perf_reg(Fixed_N,Moving_R2):
    
    Reg_GH=np.zeros((3,1))
    Reg_GH[0]=nrmse(Fixed_N,Moving_R2)
    Reg_GH[1]=nmi(Fixed_N,Moving_R2)
    Reg_GH[2]=ssim(Fixed_N,Moving_R2)
    return Reg_GH

#%% Warp cube
def warp_flt_img_3D(warp_matrix,sz_fixed,Moving_sitk_1):
    # Moving_sitk_registered=np.zeros_like(Moving_sitk)
    Moving_sitk=cv2.resize(Moving_sitk_1, (sz_fixed[1],sz_fixed[0]), interpolation= cv2.INTER_NEAREST)
    sz_moving=Moving_sitk.shape
    Moving_datatype=str(Moving_sitk_1.dtype)
    Moving_sitk_registered=np.zeros((sz_fixed[0],sz_fixed[1],sz_moving[2]))
    for page in range(sz_moving[2]):
        Moving_sitk_registered[:,:,page] = cv2.warpAffine(Moving_sitk[:,:,page], warp_matrix, (sz_fixed[1],sz_fixed[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    Moving_sitk_registered=Moving_sitk_registered.astype(Moving_datatype)
    return Moving_sitk_registered 
#%%
def coreg_hist_int_flt(base_dir):
    file_extension_type = ('.tif',) # , '.exe', 'jpg', '...')
    for hist_file in os.listdir(base_dir):
        if hist_file.endswith(file_extension_type) and hist_file.startswith('R'):
            print("Found a file {}".format(hist_file)) 
            hist_img=cv2.imread(f"{base_dir}/{hist_file}")
            
        # else:
            # print("File with the name was not found") 

    if not 'hist_img' in locals():
        sys.exit("Execution was stopped due to Hist Image file was not found error")

    ##%% Coregistration

    # Hist Image parameters
    hist_img_shape=hist_img.shape
    pix_x_hist=0.22
    pix_x_hist=0.22

    ##%% Hist Image pre-processing for registration
    thresh=200

    ##%% Mask applied
    hist_img_hsv_f,hist_img_f,hist_img_gray_f,hist_mask,hist_img_gray=coreg_img_pre_process(hist_img,thresh)

    ##%% Stitched core mat file loading

    core_mat_cont_file=base_dir+'/core_stitched.mat'
    # core_mat_contents=h5py.File(core_mat_cont_file,'r+')
    core_mat_contents=loadmat(core_mat_cont_file)
    core_mat_contents_list=list(core_mat_contents.keys())

    stitch_intensity_ref=core_mat_contents['stitch_intensity']
    stitch_intensity=stitch_intensity_ref[()]

    stitch_intensity_cube_ref=core_mat_contents['stitch_intensity_cube']
    stitch_intensity_cube=stitch_intensity_cube_ref[()]

    stitch_flt_cube_ref=core_mat_contents['stitch_flt_cube']
    stitch_flt_cube=stitch_flt_cube_ref[()]


    ##%% Saturation - Hist Registration
    Fixed=stitch_intensity
    Moving=hist_img_hsv_f[:,:,1]
    NofFeaturs=1000
    NofIterations=10000

    Fixed_N, Moving_N=prepare_img_4_reg_Fixed_changedatatype(Fixed,Moving)

    tic = time.perf_counter()
    Moving_R3, warp_matrix, cc=Affine_OpCV_2D(Fixed_N,Moving_N,NofIterations)
    toc = time.perf_counter()
    Affine_time=(toc-tic)/60
    print('Affine: %s'%Affine_time)
    mdict={'warp_matrix':warp_matrix}
    savemat(f"{base_dir}/warp_matrix.mat", mdict)

    ##%% Registration Evaluation

    Reg_SH=perf_reg(Fixed_N,Moving_R3)

    ##%% Co-registration for the whole core - hyperspectral image cube

    # Need to apply mask for 3D
    sz_fixed=Fixed_N.shape
    Moving_sitk_int=stitch_intensity_cube
    Moving_sitk_registered_int=stitch_intensity_cube
    # Moving_sitk_registered_int=cr.warp_flt_img_3D(warp_matrix,sz_fixed,Moving_sitk_int)
    Moving_sitk_flt=stitch_flt_cube
    Moving_sitk_registered_flt=stitch_flt_cube
    # Moving_sitk_registered_flt=cr.warp_flt_img_3D(warp_matrix,sz_fixed,Moving_sitk_flt)


    ##%%
    Moving_sitk=hist_img_f
    Moving_R4=warp_flt_img_3D(warp_matrix,sz_fixed,Moving_sitk)
    cv2.imwrite(f"{base_dir}/hist_registered.tiff", Moving_R4)
#%%
