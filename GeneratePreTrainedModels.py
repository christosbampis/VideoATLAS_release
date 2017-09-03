###############################################################################################
#
# Author: Christos Bampis
# Video ATLAS demo: train your own model on the whole LIVE-Netflix (or Waterloo) dataset
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
from sklearn.grid_search import GridSearchCV
import copy
from sklearn import preprocessing
from sklearn.externals import joblib
import pickle
from sklearn.svm import SVR
import sys

# read LIVE-Netflix database files
train_db = 'LIVE_NFLX'

is_windows = hasattr(sys, 'getwindowsversion')

bracket = "//"
if is_windows:
    bracket = "\\"

db_path = os.getcwd() + bracket + train_db + '_PublicData_VideoATLAS_Release' + bracket
db_files = os.listdir(db_path)

# folder to store the models
save_p = os.getcwd() + bracket + 'PretrainedModels' + bracket + train_db + bracket

# ensure nice sorting, i.e. contents 1 to 14, patterns 0 to 7    
db_files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)]) 

Nvideos = len(db_files)

# initialize data
X = np.zeros((len(db_files), 5))
y = np.zeros((len(db_files), 1))

# pick SVR regressor (or you can try RandomForest)
# regressor_name = "RandomForest"
regressor_name = "SVR"

# setup RF
param_grid_rf = {'n_estimators': [5, 10, 15, 20, 50, 100]}
cv_folds = 10

if regressor_name == "RandomForest":
    regressor = GridSearchCV(RandomForestRegressor(), param_grid_rf, cv=cv_folds)
else:
    regressor_name = "SVR"
    regressor = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=cv_folds, param_grid={"C": [1e-1, 1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 15)})

# pooling type for frame quality scores, can be mean, hyst or kmeans (for VQpooling)
pooling_type ="mean"

# pick quality model, e.g. STRRED (can be also SSIM, NIQE, MSSIM, GMSD, VMAF, PSNR)
quality_model= "STRRED"

# preprocessing switch, i.e. true or false (zero mean and unit variance, use only training data for the distribution)
preproc = True

# rename features to be consistent with paper
feat_names = ["VQA", "R$_1$", "R$_2$", "M", "I"]
feature_labels = list()

#_norm denotes that the feature is normalized with the number of frames
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
    
# use all of the dataset for training the desired regressor
X_train_before_scaling = copy.deepcopy(X)
y_train = copy.deepcopy(y)

# load a subset of the data as a sanity check on dummy test data, if needed 
# (this IS NOT the real test data)
X_test_before_scaling = X[range(20), :]
y_test = y[range(20), :]

if preproc:
    
    scaler = preprocessing.StandardScaler().fit(X_train_before_scaling)
    X_train = scaler.transform(X_train_before_scaling)

else:
                        
    X_train = copy.deepcopy(X_train_before_scaling)
    
regressor.fit(X_train, np.ravel(y_train))
if hasattr(regressor, 'best_estimator_'):
    pickle.dump([regressor.best_estimator_, scaler], open(save_p + regressor_name + "_" + quality_model + "_" + pooling_type + ".pkl", "wb"))
else:  
    pickle.dump([regressor, scaler], open(save_p + regressor_name + "_" + quality_model + "_" + pooling_type + ".pkl", "wb"))





