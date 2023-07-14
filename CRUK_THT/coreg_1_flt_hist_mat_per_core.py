#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 12:09:30 2023

@author: Arun PDRA, THT
"""

#%%
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

# import h5py
from scipy import ndimage as ndi
from scipy.io import savemat,loadmat
from coreg_lib import coreg_img_pre_process,OCV_Homography_2D,prepare_img_4_reg_Moving_changedatatype,prepare_img_4_reg_Fixed_changedatatype,Affine_OpCV_2D,perf_reg
import time