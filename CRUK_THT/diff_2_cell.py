#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 09:25:16 2023

@author: Arun PDRA, THT
"""
#%%
from scipy.io import savemat,loadmat
import numpy as np
import cv2
import matplotlib.pyplot as plt
# import sklearn
import h5py
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
#%%
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-2_20230215/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-9_20230222/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-13_20230226/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-3_Col-5_20230218/Mat_output2'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/RT/Row-4_Col-1_20230214/Mat_output2'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/RT/Row-6_Col-10_20230223/Mat_output2'


#%%
base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/'
#%%
gt_mat_cont_file=base_dir+'/train_data_Py.mat'
gt_mat_contents=h5py.File(gt_mat_cont_file,'r+')
# gt_mat_contents=loadmat(gt_mat_cont_file)
gt_mat_contents_list=list(gt_mat_contents.keys())

label_ref=gt_mat_contents['train_label']
label=label_ref[()]

feat_ref=gt_mat_contents['train_data']
feat=feat_ref[()]

del label_ref, feat_ref

#%% Data Preparation
label=np.squeeze(label)
feat=np.transpose(feat)

X_train, X_test, y_train, y_test = train_test_split(feat, label, test_size=0.2, random_state=0)

#%% Classification Model and hyperparameters
#Linear SVM
clf=svm.SVC()
# #Non Linear SVM
# clf=svm.NuSVC(gamma='auto')
# #Random Forest Tree
# clf = RandomForestClassifier(random_state=0)


#%% Train and fit
# clf.fit(feat,label)
# perf=clf.score(feat,label)

clf.fit(X_train,y_train)
perf_train=clf.score(X_train,y_train)

#%% Predictions and Confusion matrix display
predictions = clf.predict(X_test)
perf_test=clf.score(X_test,y_test)
cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
plt.show()