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
#%% Extracting information from QuPATH classification output
with open(path_to_class_file) as class_file:
    lines = [line.rstrip('\n') for line in class_file]

items = [item.split('\t') for item in lines]

del lines

column_names=items[0] # Extracting table column names

del items[0]