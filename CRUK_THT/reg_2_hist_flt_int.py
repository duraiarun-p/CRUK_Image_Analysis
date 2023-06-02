#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 13:55:37 2023

@author: Arun PDRA, THT
"""

#%%
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
import resource
resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
sys.setrecursionlimit(10**6)

import os
from os import listdir
from os.path import join, isdir
from timeit import default_timer as timer
from scipy import ndimage as ndi
import scipy
import cv2

from skimage.metrics import structural_similarity as ssim

#%%

mypath='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/Row-5_Col-11_20230224/FLT_IMG_DIR_4'
# mypath='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/Row-1_Col-9_20230222/FLT_IMG_DIR'

# onlyfiles = [join(mypath, f) for f in listdir(mypath) if isdir(join(mypath, f))]
onlyfiles = [join(mypath, f) for f in listdir(mypath)]
onlyfiles.sort()
#del onlyfiles[-1]# remove cube file from list
onlyfiles_len=len(onlyfiles)

flt_cube_list=[]
int_cube_list=[]
int_list=[]


for tile_file in range(onlyfiles_len):
    mat_fname=onlyfiles[tile_file]
    print(mat_fname)
    mat_contents = sio.loadmat(mat_fname)
    img_flt_ref=mat_contents['img_flt']
    img_int_ref=mat_contents['img_int']
    img_flt=img_flt_ref[()]
    img_int=img_int_ref[()]
    img_flt[img_flt>5]=5
    
    img_int_sl=np.sum(img_int[:,:,:],axis=-1)
    
    flt_cube_list.append(img_flt)
    int_cube_list.append(img_int)
    int_list.append(img_int_sl)
#%%
cube_siz=np.shape(flt_cube_list[0])
cube_row=cube_siz[0]*3
cube_col=cube_siz[1]*3
cube_page=cube_siz[2]

cube_flt=np.zeros((cube_row,cube_col,cube_page))

cube_int=np.zeros((cube_row,cube_col,cube_page))

tma_int=np.zeros((cube_row,cube_col))



# row_start=[0,0,0,512,512,512,1024,1024,1024]
# row_stop=[512,512,512,1024,1024,1024,1536,1536,1536]
# col_start=[0,512,1024,0,512,1024,0,512,1024]
# col_stop=[512,1024,1536,512,1024,1536,512,1024,1536]
row_start=np.array([0,0,0,512,512,512,1024,1024,1024])
row_stop=np.array([512,512,512,1024,1024,1024,1536,1536,1536])
col_start=np.array([0,512,1024,0,512,1024,0,512,1024])
col_stop=np.array([512,1024,1536,512,1024,1536,512,1024,1536])

a=np.array([1,4,7,2,5,8,3,6,9])
ai=np.argsort(a)
row_start=row_start[ai]
row_stop=row_stop[ai]
col_start=col_start[ai]
col_stop=col_stop[ai]

for tile_index in range(onlyfiles_len):
    # print('Col start:%s Col stop:%s'%(col_start[tile_index],col_stop[tile_index]))
    # print('Row start:%s Row stop:%s'%(row_start[tile_index],row_stop[tile_index]))
    
    flt_img_tile=flt_cube_list[tile_index]
    int_img_tile=int_cube_list[tile_index]
    int_list_sl=int_list[tile_index]
    
    # flt_img_tile=ndi.median_filter(flt_img_tile,size=3)
    # int_img_tile=ndi.median_filter(int_img_tile,size=3)
    # int_list_sl=ndi.median_filter(int_list_sl,size=3)
    
    
    cube_flt[row_start[tile_index]:row_stop[tile_index],col_start[tile_index]:col_stop[tile_index],:]=flt_img_tile
    cube_int[row_start[tile_index]:row_stop[tile_index],col_start[tile_index]:col_stop[tile_index],:]=int_img_tile
    tma_int[row_start[tile_index]:row_stop[tile_index],col_start[tile_index]:col_stop[tile_index]]=int_list_sl
    
    

    
cube_int=ndi.rotate(cube_int,90)
cube_flt=ndi.rotate(cube_flt, 90)

for page in range(cube_page):
    cube_int_slice=np.fliplr(cube_int[:,:,page])
    SNR=np.sqrt(np.mean(cube_int_slice))
    cube_int_slice[cube_int_slice<SNR]=0
    cube_int_slice[cube_int_slice<0]=0
    cube_int[:,:,page]=cube_int_slice
    
    cube_flt_slice=np.fliplr(cube_flt[:,:,page])
    cube_flt_slice[cube_int_slice<SNR]=0
    cube_flt_slice[cube_int_slice<0]=0
    cube_flt_slice[cube_flt_slice<0]=0
    cube_flt_slice[cube_flt_slice>5]=5
    cube_flt[:,:,page]=cube_flt_slice

tma_int=np.fliplr(ndi.rotate(tma_int, 90))

#%%

#%%
# row_start=row_start
# row_stop=row_stop
# col_start=col_start
# col_stop=col_stop[ai]
overlap_size=12
tma_int_f=np.zeros((cube_row-(6*overlap_size),cube_col-(6*overlap_size)))
cube_flt_f=np.zeros((cube_row-(6*overlap_size),cube_col-(6*overlap_size),cube_page))
cube_int_f=np.zeros((cube_row-(6*overlap_size),cube_col-(6*overlap_size),cube_page))

# for tile_index in range(onlyfiles_len):
#     row_x=max(0,row_start[tile_index]-overlap_size)
#     # row_x=min(row_start[tile_index],row_start[tile_index]-overlap_size)
#     # row_X=min(row_stop[tile_index],row_stop[tile_index]-overlap_size)
#     row_X=row_stop[tile_index]-overlap_size
#     col_x=max(0,col_start[tile_index]-overlap_size)
#     # col_x=min(col_start[tile_index],col_start[tile_index]-overlap_size)
#     # col_X=min(col_stop[tile_index],col_stop[tile_index]-overlap_size)
#     col_X=col_stop[tile_index]-overlap_size
    
#     print('Col start:%s Col stop:%s'%(col_x,col_X))
#     print('Row start:%s Row stop:%s'%(row_x,row_X))
    
#     print('Col start:%s Col stop:%s'%(col_start[tile_index],col_stop[tile_index]))
#     print('Row start:%s Row stop:%s'%(row_start[tile_index],row_stop[tile_index]))
    

#     # tma_int_f[row_x:row_X,col_x:col_X]=tma_int[row_start[tile_index]:row_stop[tile_index],col_start[tile_index]:col_stop[tile_index]]
    
#     tma_int_f[row_x:row_X,col_x:col_X]=tma_int[row_start[tile_index]:row_stop[tile_index]-overlap_size,col_start[tile_index]:col_stop[tile_index]-overlap_size]




# tma_int_f[0:488,0:488]=tma_int[0:488,0:488]
# tma_int_f[0:488,488:976]=tma_int[0:488,512:1000]
# tma_int_f[0:488,976:1464]=tma_int[0:488,1024:1512]

#         # tma_int_f[488:976,0:488]=tma_int[512:1000,0:488]
#         # tma_int_f[488:976,488:976]=tma_int[512:1000,512:1000]
#         # tma_int_f[488:976,976:1464]=tma_int[512:1000,1024:1512]

# tma_int_f[488:976,0:488]=tma_int[536:1024,0:488]
# tma_int_f[488:976,488:976]=tma_int[536:1024,512:1000]
# tma_int_f[488:976,976:1464]=tma_int[536:1024,1024:1512]

#             # tma_int_f[976:1464,0:488]=tma_int[1048:1536,0:488]
#             # tma_int_f[976:1464,488:976]=tma_int[1048:1536,512:1000]
#             # tma_int_f[976:1464,976:1464]=tma_int[1048:1536,1024:1512]

# tma_int_f[976:1440,0:488]=tma_int[1072:1536,0:488]
# tma_int_f[976:1440,488:976]=tma_int[1072:1536,512:1000]
# tma_int_f[976:1440,976:1464]=tma_int[1072:1536,1024:1512]


# row_step=cube_siz[0]-(2*overlap_size)
# col_step=cube_siz[1]-(2*overlap_size)



def img_tiling(tma_intx,tma_int_fx):
    
    if len(tma_intx.shape)>2:
        tma_int_fx[0:488,0:488,:]=tma_intx[0:488,0:488,:]
        tma_int_fx[0:488,488:976,:]=tma_intx[0:488,512:1000,:]
        tma_int_fx[0:488,976:1464,:]=tma_intx[0:488,1024:1512,:]

        tma_int_fx[488:976,0:488,:]=tma_intx[536:1024,0:488,:]
        tma_int_fx[488:976,488:976,:]=tma_intx[536:1024,512:1000,:]
        tma_int_fx[488:976,976:1464,:]=tma_intx[536:1024,1024:1512,:]

        tma_int_fx[976:1440,0:488,:]=tma_intx[1072:1536,0:488,:]
        tma_int_fx[976:1440,488:976,:]=tma_intx[1072:1536,512:1000,:]
        tma_int_fx[976:1440,976:1464,:]=tma_intx[1072:1536,1024:1512,:]
    else:
        tma_int_fx[0:488,0:488]=tma_intx[0:488,0:488]
        tma_int_fx[0:488,488:976]=tma_intx[0:488,512:1000]
        tma_int_fx[0:488,976:1464]=tma_intx[0:488,1024:1512]

        tma_int_fx[488:976,0:488]=tma_intx[536:1024,0:488]
        tma_int_fx[488:976,488:976]=tma_intx[536:1024,512:1000]
        tma_int_fx[488:976,976:1464]=tma_intx[536:1024,1024:1512]

        tma_int_fx[976:1440,0:488]=tma_intx[1072:1536,0:488]
        tma_int_fx[976:1440,488:976]=tma_intx[1072:1536,512:1000]
        tma_int_fx[976:1440,976:1464]=tma_intx[1072:1536,1024:1512]
    return tma_int_fx

def img_tile_remove(tma_int_fx):
    tma_int_fx[:,487]=(tma_int_fx[:,488]+tma_int_fx[:,489])*0.5
    tma_int_fx[:,975]=(tma_int_fx[:,974]+tma_int_fx[:,976])*0.5
    tma_int_fx[487,:]=(tma_int_fx[488,:]+tma_int_fx[489,:])*0.5
    tma_int_fx[975,:]=(tma_int_fx[974,:]+tma_int_fx[976,:])*0.5
    
    # tma_fx_arr=
    return tma_int_fx

#%%


tma_int_f=img_tiling(tma_int,tma_int_f)
sz=tma_int_f.shape

tma_int_f_N = np.zeros_like(tma_int_f)
tma_int_f_N = np.round(cv2.normalize(tma_int_f,  tma_int_f_N, 0, 255, cv2.NORM_MINMAX))
tma_int_f_N=tma_int_f_N.astype('uint8')

#%%



import SimpleITK as sitk


def Rigid_SITK_2(Fixed,Moving,FixedSpacing,MovingSpacing):    
    Fixed_sitk=sitk.GetImageFromArray(Fixed)
    Moving_sitk=sitk.GetImageFromArray(Moving)
    Fixed_sitk=sitk.Cast(Fixed_sitk,sitk.sitkFloat64)
    Moving_sitk=sitk.Cast(Moving_sitk,sitk.sitkFloat64)
    Fixed_sitk.SetSpacing([FixedSpacing[0,2],FixedSpacing[0,1],FixedSpacing[0,0]])
    Moving_sitk.SetSpacing([MovingSpacing[0,2],MovingSpacing[0,1],MovingSpacing[0,0]])
    initial_transform = sitk.CenteredTransformInitializer(Fixed_sitk,  
                                                          Moving_sitk, 
                                                          sitk.Euler3DTransform(), 
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=1000)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    # registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
    registration_method.SetMetricSamplingPercentage(0.01)
    
    # registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetInterpolator(sitk.sitkBSpline)
    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, 
                                                        numberOfIterations=50,
                                                        convergenceMinimumValue=1e-6, 
                                                        convergenceWindowSize=10)
    
    # registration_method.SetOptimizerAsOnePlusOneEvolutionaryOptimizerv4(Growthfactor=1.0500,
    #                                                                     Epsilon=1.5000e-06,
    #                                                                     InitialRadius=0.0063,
    #                                                                     MaximumIteration=100)
    
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    # Setup for the multi-resolution framework.            
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # Set the initial moving and optimized transforms.
    optimized_transform = sitk.Euler3DTransform()    
    registration_method.SetMovingInitialTransform(initial_transform)
    registration_method.SetInitialTransform(optimized_transform, inPlace=True)
    
    
    final_transform_v4 = registration_method.Execute(Fixed_sitk, Moving_sitk)
    
    Moving_sitk_registered = sitk.Resample(Moving_sitk, Fixed_sitk, 
                                                  final_transform_v4, 
                                                  sitk.sitkBSpline, 0.0, 
                                                  Moving_sitk.GetPixelID())
    
    
    Moving_sitk_registered=sitk.GetArrayFromImage(Moving_sitk_registered)
    # Moving_sitk_registered=np.transpose(Moving_sitk_registered,(1,2,0))
    
    return Moving_sitk_registered,registration_method,final_transform_v4

def Rigid_SITK_2D(Fixed,Moving,FixedSpacing,MovingSpacing):    
    Fixed_sitk=sitk.GetImageFromArray(Fixed)
    Moving_sitk=sitk.GetImageFromArray(Moving)
    Fixed_sitk=sitk.Cast(Fixed_sitk,sitk.sitkFloat64)
    Moving_sitk=sitk.Cast(Moving_sitk,sitk.sitkFloat64)
    # Fixed_sitk.SetSpacing([FixedSpacing[0,2],FixedSpacing[0,1],FixedSpacing[0,0]])
    # Moving_sitk.SetSpacing([MovingSpacing[0,2],MovingSpacing[0,1],MovingSpacing[0,0]])
    Fixed_sitk.SetSpacing([FixedSpacing[1],FixedSpacing[0]])
    Moving_sitk.SetSpacing([MovingSpacing[1],MovingSpacing[0]])
    initial_transform = sitk.CenteredTransformInitializer(Fixed_sitk,  
                                                          Moving_sitk, 
                                                          sitk.Euler2DTransform(), 
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    
    # initial_transform = sitk.CenteredTransformInitializer(sitk.Cast(Fixed_sitk,  
    #                                                       Moving_sitk.GetPixelID()), 
    #                                                       sitk.Euler2DTransform(), 
    #                                                       sitk.CenteredTransformInitializerFilter.GEOMETRY)
    
    registration_method = sitk.ImageRegistrationMethod()
    # registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricAsCorrelation()
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    # registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
    registration_method.SetMetricSamplingPercentage(0.01)
    
    registration_method.SetInterpolator(sitk.sitkLinear)
    # registration_method.SetInterpolator(sitk.sitkBSpline)
    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, 
                                                        numberOfIterations=1000,
                                                        convergenceMinimumValue=1e-6, 
                                                        convergenceWindowSize=10)
    
    # registration_method.SetOptimizerAsOnePlusOneEvolutionaryOptimizerv4(Growthfactor=1.0500,
    #                                                                     Epsilon=1.5000e-06,
    #                                                                     InitialRadius=0.0063,
    #                                                                     MaximumIteration=100)
    
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    # Setup for the multi-resolution framework.            
    # registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    # registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    # registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # Set the initial moving and optimized transforms.
    optimized_transform = sitk.Euler2DTransform()    
    registration_method.SetMovingInitialTransform(initial_transform)
    registration_method.SetInitialTransform(optimized_transform, inPlace=True)
    
    
    final_transform_v4 = registration_method.Execute(Fixed_sitk, Moving_sitk)
    
    # Moving_sitk_registered = sitk.Resample(Moving_sitk, Fixed_sitk, 
    #                                               final_transform_v4, 
    #                                               sitk.sitkBSpline, 0.0, 
    #                                               Moving_sitk.GetPixelID())
    
    Moving_sitk_registered = sitk.Resample(Moving_sitk, Fixed_sitk, 
                                                  final_transform_v4, 
                                                  sitk.sitkLinear, 0.0, 
                                                  Moving_sitk.GetPixelID())
    
    
    Moving_sitk_registered=sitk.GetArrayFromImage(Moving_sitk_registered)
    # Moving_sitk_registered=np.transpose(Moving_sitk_registered,(1,2,0))
    
    return Moving_sitk_registered,registration_method,final_transform_v4

def Affine_SITK_2D(Fixed,Moving,FixedSpacing,MovingSpacing):    
    Fixed_sitk=sitk.GetImageFromArray(Fixed)
    Moving_sitk=sitk.GetImageFromArray(Moving)
    Fixed_sitk=sitk.Cast(Fixed_sitk,sitk.sitkFloat64)
    Moving_sitk=sitk.Cast(Moving_sitk,sitk.sitkFloat64)
    # Fixed_sitk.SetSpacing([FixedSpacing[0,2],FixedSpacing[0,1],FixedSpacing[0,0]])
    # Moving_sitk.SetSpacing([MovingSpacing[0,2],MovingSpacing[0,1],MovingSpacing[0,0]])
    Fixed_sitk.SetSpacing([FixedSpacing[1],FixedSpacing[0]])
    Moving_sitk.SetSpacing([MovingSpacing[1],MovingSpacing[0]])
    initial_transform = sitk.CenteredTransformInitializer(Fixed_sitk,  
                                                          Moving_sitk, 
                                                          sitk.AffineTransform(2), 
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    
    # initial_transform = sitk.CenteredTransformInitializer(sitk.Cast(Fixed_sitk,  
    #                                                       Moving_sitk.GetPixelID()), 
    #                                                       sitk.Euler2DTransform(), 
    #                                                       sitk.CenteredTransformInitializerFilter.GEOMETRY)
    
    registration_method = sitk.ImageRegistrationMethod()
    # registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricAsCorrelation()
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    # registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
    registration_method.SetMetricSamplingPercentage(0.01)
    
    registration_method.SetInterpolator(sitk.sitkLinear)
    # registration_method.SetInterpolator(sitk.sitkBSpline)
    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, 
                                                        numberOfIterations=1000,
                                                        convergenceMinimumValue=1e-6, 
                                                        convergenceWindowSize=10)
    
    # registration_method.SetOptimizerAsOnePlusOneEvolutionaryOptimizerv4(Growthfactor=1.0500,
    #                                                                     Epsilon=1.5000e-06,
    #                                                                     InitialRadius=0.0063,
    #                                                                     MaximumIteration=100)
    
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    # Setup for the multi-resolution framework.            
    # registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    # registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    # registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # Set the initial moving and optimized transforms.
    optimized_transform = sitk.AffineTransform(2)    
    registration_method.SetMovingInitialTransform(initial_transform)
    registration_method.SetInitialTransform(optimized_transform, inPlace=True)
    
    
    final_transform_v4 = registration_method.Execute(Fixed_sitk, Moving_sitk)
    
    # Moving_sitk_registered = sitk.Resample(Moving_sitk, Fixed_sitk, 
    #                                               final_transform_v4, 
    #                                               sitk.sitkBSpline, 0.0, 
    #                                               Fixed_sitk.GetPixelID())
    
    # Moving_sitk_registered = sitk.Resample(Moving_sitk, Fixed_sitk, 
    #                                               final_transform_v4, 
    #                                               sitk.sitkLinear, 0.0, 
    #                                               Moving_sitk.GetPixelID())
    
    Moving_sitk_registered = sitk.Resample(Moving_sitk, Fixed_sitk, 
                                                  final_transform_v4, 
                                                  sitk.sitkBSpline, 0.0, 
                                                  Moving_sitk.GetPixelID())
    
    
    Moving_sitk_registered=sitk.GetArrayFromImage(Moving_sitk_registered)
    # Moving_sitk_registered=np.transpose(Moving_sitk_registered,(1,2,0))
    
    return Moving_sitk_registered,registration_method,final_transform_v4

def Affine_OpCV_2D(Fixed_sitk,Moving_sitk):
    sz=Fixed_sitk.shape
    warp_mode = cv2.MOTION_AFFINE

    warp_matrix = np.eye(2, 3, dtype=np.float32)

    number_of_iterations = 50000


    termination_eps = 1e-10

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    (cc, warp_matrix) = cv2.findTransformECC (tma_int_f_N,Moving_sitk,warp_matrix, warp_mode, criteria)

    # # warp_matrix=cv2.getAffine

    Moving_sitk_registered = cv2.warpAffine(Moving_sitk, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    
    return Moving_sitk_registered, warp_matrix, cc

#%%




flt_h=sz[0]
flt_w=sz[1]
tma_scan =  cv2.imread("/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/Row-5_Col-11_20230224/Row-5_Col-11.tif")



cx_p=0.22 # Unit: micrometer
cy_p=0.22

# px=95 # Unit: micrometer
# py=95
phy=tma_scan.shape
phy_x=phy[0]*cx_p
phy_y=phy[1]*cy_p

px_p=phy_x/flt_h
py_p=phy_x/flt_w

cx=1
cy=1
px=1
py=1


tma_scan=cv2.resize(tma_scan, (flt_h,flt_w))

# tma_scan[tma_scan==255]=0

# tma_scan_f=cv2.bitwise_not(cv2.cvtColor(tma_scan,cv2.COLOR_BGR2GRAY))
# tma_scan_f=(cv2.cvtColor(tma_scan,cv2.COLOR_BGR2GRAY))

tma_scan_hsv = cv2.cvtColor(tma_scan, cv2.COLOR_BGR2HSV)
tma_scan_f=tma_scan_hsv[:,:,2]
tma_scan_f[tma_scan_f>200]=0


Fixed=tma_int_f_N
Moving=tma_scan_f
# Fixed=Moving
FixedSpacing=[cx,cy]
MovingSpacing=[px_p,py_p]
MovingSpacing=[px,py]

# Moving=tma_int_f_N
# Fixed=tma_scan_f
# MovingSpacing=[cx,cy]
# FixedSpacing=[px,py]

Moving_R,registration_method,final_transform_v4=Rigid_SITK_2D(Fixed,Moving,FixedSpacing,MovingSpacing)
Moving_R=Moving_R.astype('uint8')
DIff=np.abs(Moving_R-Moving)
RE=np.sum(DIff)
(RE,DIff1)=ssim(Fixed, Moving_R, full=True)
# (RE,DIff1)=ssim(Fixed, Moving_R, full=False)
plt.figure(1),
plt.subplot(2,2,1)
plt.imshow(Moving,cmap='gray')
plt.subplot(2,2,2)
plt.imshow(Fixed,cmap='gray')
plt.subplot(2,2,3)
plt.imshow(Moving_R,cmap='gray')
plt.subplot(2,2,4)
plt.imshow(DIff,cmap='gray')
plt.title(RE)
plt.show()


Moving_R,registration_method,final_transform_v4=Affine_SITK_2D(Fixed,Moving,FixedSpacing,MovingSpacing)
Moving_R=Moving_R.astype('uint8')
DIff=np.abs(Moving_R-Moving)
RE=np.sum(DIff)
(RE,DIff1)=ssim(Fixed, Moving_R, full=True)
# (RE,DIff1)=ssim(Fixed, Moving_R, full=False)
plt.figure(2),
plt.subplot(2,2,1)
plt.imshow(Moving,cmap='gray')
plt.subplot(2,2,2)
plt.imshow(Fixed,cmap='gray')
plt.subplot(2,2,3)
plt.imshow(Moving_R,cmap='gray')
plt.subplot(2,2,4)
plt.imshow(DIff,cmap='gray')
plt.title(RE)
plt.show()


# Error would just show the displacements.
# SSIM would give the closeness to the Fixed Image considering from different modality


Moving_R, warp_matrix, cc=Affine_OpCV_2D(Fixed,Moving)
Moving_R=Moving_R.astype('uint8')
DIff=np.abs(Moving_R-Moving)
RE=np.sum(DIff)
(RE,DIff1)=ssim(Fixed, Moving_R, full=True)
# (RE,DIff1)=ssim(Fixed, Moving_R, full=False)
plt.figure(3),
plt.subplot(2,2,1)
plt.imshow(Moving,cmap='gray')
plt.subplot(2,2,2)
plt.imshow(Fixed,cmap='gray')
plt.subplot(2,2,3)
plt.imshow(Moving_R,cmap='gray')
plt.subplot(2,2,4)
plt.imshow(DIff,cmap='gray')
plt.title(RE)
plt.show()

#%%
# plt.figure(1),
# plt.subplot(1,4,1)
# plt.imshow(Moving,cmap='gray')
# plt.subplot(1,4,2)
# plt.imshow(Fixed,cmap='gray')
# plt.subplot(1,4,3)
# plt.imshow(Moving_R,cmap='gray')
# plt.subplot(1,4,4)
# plt.imshow(DIff,cmap='gray')
# plt.show()

