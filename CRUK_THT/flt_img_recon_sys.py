#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 09:37:30 2023

@author: Arun PDRA, THT
"""
#%% Importing Libraries
import sys
import getopt
from os import listdir
from os.path import join, isdir
from timeit import default_timer as timer

import resource
resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
sys.setrecursionlimit(10**6)


from flt_img_est_lib import flt_per_tile
from scipy.io import savemat

def main(argv):
        
    # argv=getopt.getopt(arg)

    # # total arguments
    # n = len(sys.argv)
    # print("Total arguments passed:", n)
     
    # # Arguments passed
    # print("\n 1st :", sys.argv[0])
    # print("\n 2nd :", sys.argv[1])
    # print("\n 3rd :", sys.argv[2])
    
    # total arguments
    n = len(argv)
    print("Total arguments passed:", n)
     
    # Arguments passed
    print("\n 1st :", argv[0])
    print("\n 2nd :", argv[1])
    print("\n 3rd :", argv[2])
    
    #%% Path setting
    
    # mypath=sys.argv[1]
    # save_path=sys.argv[2]
    
    mypath=argv[1]
    save_path=argv[2]
    
    mypath_splitted = mypath.replace('/',',')
    mypath_splitted = mypath_splitted.split(',')
    tissue_core_file_name=mypath_splitted[-2]
    print(tissue_core_file_name)
    
    decimate_factor=15
    spec_resampled=20# Change it to 20
    spec_truncated=330
    # mypath = '/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour/Row-1_Col-1_20230214'
    # List of responses of all tiles
    onlyfiles = [join(mypath, f) for f in listdir(mypath) if isdir(join(mypath, f))]
    onlyfiles.sort()
    del onlyfiles[0]# remove first directory index - ouput directory
    onlyfiles_len=len(onlyfiles)
    
    #%% Image Reconstruction
    
    start_time_0_I=timer()
    # onlyfiles_len=1
    
    for tile_file in range(onlyfiles_len):
        
    
        matfile_list=listdir(onlyfiles[tile_file])
        #iterable tile
        matfile_list_path=join(onlyfiles[tile_file],matfile_list[0])#picking the mat file
        
        start_time_0=timer()
        img_int,img_flt,wave_spectrum,wave_spectrum_new,wave_spectrum_new_select=flt_per_tile(matfile_list_path,decimate_factor,spec_resampled,spec_truncated)
        runtimeN3=(timer()-start_time_0)/60
        mdic = {"img_int": img_int, "img_flt":img_flt, "wave_spectrum":wave_spectrum,"wave_spectrum_new":wave_spectrum_new,"runtimeN3":runtimeN3}
        # mdic ={"tile_file":tile_file}
        print('Tile built time %s'%runtimeN3)
        matfile_filename=save_path+'/'+tissue_core_file_name+'-'+str(tile_file)+'.mat'
        print(matfile_filename)
        savemat(matfile_filename, mdic)
    
    runtimeN3=(timer()-start_time_0_I)/60
    print('All Tiles built time %s'%runtimeN3)
#%%
if __name__=='__main__':
    main(sys.argv)
    # print(sys.argv)