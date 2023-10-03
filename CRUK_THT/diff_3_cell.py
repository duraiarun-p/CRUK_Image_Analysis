#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 21:36:58 2023

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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,balanced_accuracy_score,f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import pickle

from timeit import default_timer as timer
#%%
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-2_20230215/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-9_20230222/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-1_Col-13_20230226/Mat_output'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/Row-3_Col-5_20230218/Mat_output2'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/RT/Row-4_Col-1_20230214/Mat_output2'
# base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/RT/Row-6_Col-10_20230223/Mat_output2'


#%%
model_base_dir='/home/cruk/Documents/PyWS_CRUK/CRUK_Image_Analysis/Test_Data/Tumour_1/'
#%%
gt_mat_cont_file=model_base_dir+'/train_data_Py_Dist_1to4.mat'

gt_mat_contents=h5py.File(gt_mat_cont_file,'r')
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

X_train, X_test, y_train, y_test = train_test_split(feat, label, test_size=0.33, random_state=10)

#%% Classification Model and hyperparameters
# #Linear SVM
# clf=svm.SVC()
# #Non Linear SVM
# clf=svm.NuSVC(gamma='auto',decision_function_shape='ovo')
#Random Forest Tree
clf = RandomForestClassifier(random_state=0)

model_filename = model_base_dir+'NuSVM_CV_trained.sav'

#%% Train and fit
# clf.fit(feat,label)
# perf=clf.score(feat,label)
start_time_0_I=timer()

# clf.fit(X_train,y_train)
# perf_train=clf.score(X_train,y_train)

#code checked and tested

scores=[]
cv_fold=10
kFold=KFold(n_splits=cv_fold,random_state=42,shuffle=True)
# for train_index,test_index in kFold.split(feat):
#     # print("Train Index: ", train_index, "\n")
#     # print("Test Index: ", test_index)   
#     X_train_1, X_test_1, y_train_1, y_test_1 = feat[train_index,:], feat[test_index,:], label[train_index], label[test_index]
#     clf.fit(X_train_1, y_train_1)
#     scores.append(clf.score(X_test_1, y_test_1))

for train_index,test_index in kFold.split(X_train):
    # print("Train Index: ", train_index, "\n")
    # print("Test Index: ", test_index)   
    X_train_1, X_test_1, y_train_1, y_test_1 = X_train[train_index,:], X_train[test_index,:], y_train[train_index], y_train[test_index]
    clf.fit(X_train_1, y_train_1)
    scores.append(clf.score(X_test_1, y_test_1))

perf_cv=[np.mean(scores),np.std(scores)]
# perf_cv_std=np.std(scores)
# print(perf_cv)


predictions_train = clf.predict(X_train)
perf_train=[balanced_accuracy_score(y_train, predictions_train),f1_score(y_train, predictions_train,average="micro")]
print(perf_train)

#%% Predictions and Confusion matrix display
predictions = clf.predict(X_test)
perf_test=[balanced_accuracy_score(y_test, predictions),f1_score(y_test, predictions,average="micro")]
# print(perf_test)
perf_mcc=matthews_corrcoef(y_test, predictions)

runtimeN3=(timer()-start_time_0_I)/60
print('Training time %s'%runtimeN3)

cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
plt.show()
#%%

clf_score=[perf_train[0],perf_test[0],perf_cv[0],perf_cv[1],perf_train[1],perf_test[1],perf_mcc]
print('Acc-train,Acc-test,CV-mu,CV-std,F1-train,F1-test,MCC')
print(clf_score)

#%%
# pickle.dump(clf, open(model_filename, 'wb'))