#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 08:34:24 2023

@author: Arun PDRA, THT
"""

#%%
from scipy.io import savemat,loadmat
import numpy as np
# import cv2
import matplotlib.pyplot as plt
# import sklearn
import h5py
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,balanced_accuracy_score,f1_score
from sklearn.model_selection import train_test_split

import pickle

from timeit import default_timer as timer
#%%
base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-2_20230215/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-9_20230222/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-13_20230226/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-3_Col-5_20230218/Mat_output2'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/RT/Row-4_Col-1_20230214/Mat_output2'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/RT/Row-6_Col-10_20230223/Mat_output2'

model_base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/'
#%%


gt_mat_cont_file=base_dir+'/feat_flt_all.mat'
gt_mat_contents=h5py.File(gt_mat_cont_file,'r')
# gt_mat_contents=loadmat(gt_mat_cont_file)
gt_mat_contents_list=list(gt_mat_contents.keys())

label_ref=gt_mat_contents['class_grnd_trth']
y_test=label_ref[()]

feat_ref=gt_mat_contents['feat_matrix']
X_test=feat_ref[()]

del label_ref, feat_ref


#%% Data Preparation
y_test=np.squeeze(y_test)
feat=np.transpose(X_test)

#%% Classification model
# model_filename = model_base_dir+'NuSVM_CV_trained.sav'
model_filename = model_base_dir+'RFT_CV_trained_1.sav'

loaded_model = pickle.load(open(model_filename, 'rb'))
# result = loaded_model.score(feat, label)

start_time_0_I=timer()

predictions = loaded_model.predict(X_test)
perf_test=[balanced_accuracy_score(y_test, predictions),f1_score(y_test, predictions,average="micro")]
print(perf_test)

runtimeN3=(timer()-start_time_0_I)/60
print('Training and Testing time %s'%runtimeN3)

cm = confusion_matrix(y_test, predictions, labels=loaded_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=loaded_model.classes_)
disp.plot()
plt.show()