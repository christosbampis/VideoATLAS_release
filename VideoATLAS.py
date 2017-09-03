###############################################################################################
#
# Author: Christos Bampis
# Video ATLAS demo: train and test on a single 80% train and 20% test content split on the LIVE-Netflix dataset
#
# References:
# 1) C. G. Bampis and A. C. Bovik, "Video ATLAS Software Release" 
# URL: http://live.ece.utexas.edu/research/Quality/VideoATLAS_release.zip, 2016
# 2) C. G. Bampis and A. C. Bovik, "Learning to Predict Streaming Video QoE: Distortions, Rebuffering and Memory," under review
#
###############################################################################################

import scipy.io as sio
import os
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
import copy
from sklearn import preprocessing
from scipy.stats import spearmanr as sr
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import sys

train_db = 'LIVE_NFLX'
test_db = 'LIVE_NFLX'

is_windows = hasattr(sys, 'getwindowsversion')

bracket = "//"
if is_windows:
    bracket = "\\"

# use preloaded model trained on the whole LIVE-NFLX dataset (if it doesn't exist you can generate
# it using the GeneratePreTrainedModels.py script)
# note: if you use the pre-loaded models and then test on LIVE-NFLX, then you will overfit:
# the models have been trained on the whole dataset
preload = True

# preprocessing switch, i.e. true or false (zero mean and unit variance, use only training data for the distribution)
# note: if you pre-process the features and use a pre-loaded model, then the scaler object
# (which was computed on the whole LIVE-NFLX dataset) is also pre-loaded, else it is computed on the current training data
preproc = True

# pick quality model, e.g. MSSIM (can be also SSIM, NIQE, SSIM, GMSD, VMAF, PSNR_oss, PSNRhvs_oss)
quality_model= "STRRED"
    
# pooling type for frame quality scores, can be mean, hyst or kmeans (for VQpooling)
pooling_type = "mean"

if preload:
    
    # folders where the preloaded models are stored
    load_p = os.getcwd() + bracket + 'PretrainedModels' + bracket + train_db + bracket

    #regressor name (e.g. RanfomForest or SVR)
    regressor_name = "SVR"
    #regressor_name = "RandomForest"
    
    # select type of model, e.g. RandomForest_SSIM_mean
    # the first word should be the type of the regressor, the second the quality model used
    # and the third the pooling that has been applied on the quality model

    model_to_preload = regressor_name + "_" + quality_model + "_" + pooling_type

else:
    
    # setup the SVR regressor or use a different one (e.g. Random Forest)
    cv_folds = 10
    
    regressor_name = "SVR"
    regressor = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=cv_folds, param_grid={"C": [1e-1, 1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 15)})
    
    #param_grid_rf = {'n_estimators': [5, 10, 15, 20, 50, 100]}   
    #regressor = GridSearchCV(RandomForestRegressor(), param_grid_rf, cv=cv_folds)

# read LIVE-Netflix database files
db_path = os.getcwd() + bracket + 'LIVE_NFLX_PublicData_VideoATLAS_Release' + bracket
db_files = os.listdir(db_path)

# ensure nice sorting, i.e. contents 1 to 14, patterns 0 to 7    
db_files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)]) 

Nvideos = len(db_files)

# load 1000 train/test content splits: a value of 1 means that the ith sequence is a training sequence for the jth trial
pre_load_train_test_data_LIVE_Netflix = sio.loadmat("TrainingMatrix_LIVENetflix_1000_trials.mat")["TrainingMatrix_LIVENetflix_1000_trials"]

# randomly pick a trial out of the 1000
nt_rand = np.random.choice(np.shape(pre_load_train_test_data_LIVE_Netflix)[1], 1)
n_train = [ind for ind in range(0, Nvideos) if pre_load_train_test_data_LIVE_Netflix[ind, nt_rand] == 1]
n_test = [ind for ind in range(0, Nvideos) if pre_load_train_test_data_LIVE_Netflix[ind, nt_rand] == 0]

feat_names = ["VQA", "R$_1$", "R$_2$", "M", "I"]

# initialize data
X = np.zeros((len(db_files), len(feat_names)))
y = np.zeros((len(db_files), 1))

# rename features to be consistent with paper
feature_labels = list()

for typ in feat_names:
    if typ == "VQA":
        feature_labels.append(quality_model + "_" + pooling_type)
    elif typ == "R$_1$":
        feature_labels.append("ds_norm")
    elif typ == "R$_2$":
        feature_labels.append("ns")
    elif typ == "M":
        feature_labels.append("tsl_norm") 
    else:
        feature_labels.append("lt_norm")    

# extract db files and save as features    
for i, f in enumerate(db_files):
    data = sio.loadmat(db_path + f)
    for feat_cnt, feat in enumerate(feature_labels):
        X[i, feat_cnt] = data[feat]
    y[i] = data["final_subj_score"]   
    
# do train/test split based on n_traina nd n_test indices
X_train_before_scaling = X[n_train, :]   
X_test_before_scaling = X[n_test, :]
y_train = y[n_train]   
y_test = y[n_test]    

if preload:
    
    regressor = joblib.load(load_p + model_to_preload + ".pkl")[0]
    scaler = joblib.load(load_p + model_to_preload + ".pkl")[1]
    
    if preproc:
        
        X_train = scaler.transform(X_train_before_scaling)
        X_test = scaler.transform(X_test_before_scaling)
    
    else:
                            
        X_train = copy.deepcopy(X_train_before_scaling)
        X_test = copy.deepcopy(X_test_before_scaling)      
    
else:
    
    if preproc:
        
        scaler = preprocessing.StandardScaler().fit(X_train_before_scaling)
        X_train = scaler.transform(X_train_before_scaling)
        X_test = scaler.transform(X_test_before_scaling)
    
    else:
                            
        X_train = copy.deepcopy(X_train_before_scaling)
        X_test = copy.deepcopy(X_test_before_scaling)    
    
    regressor.fit(X_train, np.ravel(y_train))
    
if hasattr(regressor, 'best_estimator_'):
    answer = regressor.best_estimator_.predict(X_test)
else:
    answer = regressor.predict(X_test)    
    
# locate column of quality model for SROCC without regression    
if quality_model + "_" + pooling_type in feature_labels:
    position_vqa = feature_labels.index(quality_model + "_" + pooling_type) 
    
# display results     
plt.figure()
ax1 = plt.subplot(1,1,1)
plt.title("before: " + format(sr(y_test, X_test[:, position_vqa].reshape(-1,1))[0], '.4f'))
plt.scatter(y_test, X_test_before_scaling[:, position_vqa].reshape(-1,1))
plt.grid()
x0,x1 = ax1.get_xlim()
y0,y1 = ax1.get_ylim()
ax1.set_aspect((x1-x0)/(y1-y0))
plt.ylabel("predicted QoE")
plt.xlabel("MOS")
plt.show()
     
plt.figure()
ax1 = plt.subplot(1,1,1)
plt.title("after: " + format(sr(y_test, answer.reshape(-1,1))[0], '.4f'))
plt.scatter(y_test, answer.reshape(-1,1))    
plt.grid()
x0,x1 = ax1.get_xlim()
y0,y1 = ax1.get_ylim()
ax1.set_aspect((x1-x0)/(y1-y0))
plt.ylabel("predicted QoE")
plt.xlabel("MOS")    
plt.show()

print("SROCC before (" + str(quality_model) + "): " + str(sr(y_test, X_test[:, position_vqa].reshape(-1,1))[0]))
print("SROCC using VideoATLAS (" + str(quality_model) + " + " + regressor_name + "): "  + str(sr(y_test, answer.reshape(-1,1))[0]))    