#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 09:37:13 2023

@author: Arun PDRA, THT
"""

import os
import shutil


#%%

HE_dir='/mnt/local_share/TMA/histology/HE_stain'

HE_files_all=[filename for filename in os.listdir(HE_dir) if filename.startswith("R")]

FL_dirs='/mnt/local_share/TMA/FS-FLIM/raw/Tumour_2B/'

FL_files_all=os.listdir(FL_dirs)



FL_files_all_1=[]#HE list
FL_files_all_2=[]

for item in FL_files_all:
    item = item.split('_2023')
    img_item=item[0]
    FL_files_all_2.append(img_item)
    
for item in HE_files_all:
    item = item.split('.tif')
    FL_files_all_1.append(item[0])
    
core_indx_lis=[]
for item in FL_files_all_1:
    item_idx=FL_files_all_2.index(item)# Finding HE file name in FLT listdir
    core_indx_lis.append(item_idx)
    
FL_files_all_F=[]#FLT directory
for item_idx in core_indx_lis:
    FL_files_all_F.append(FL_files_all[item_idx])
    
for he,flt in zip(FL_files_all_1,FL_files_all_F):
    he_path=HE_dir+'/'+he+'.tif'
    flt_path=FL_dirs+flt+'/Mat_output2/'
    print(he_path)
    print(flt_path)
    shutil.copy2(he_path, flt_path)
    