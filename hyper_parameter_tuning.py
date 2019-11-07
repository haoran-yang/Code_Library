import pandas as pd 
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier


# 1. 随机网格搜索
def RandomSearch(clf, params, X, y, n_iter):
    cscv = RandomizedSearchCV(clf, params, n_iter=n_iter, scoring='roc_auc', n_jobs=-1, cv=5)
    cscv.fit(X, y)
    return cscv

params = {'num_leaves': 256,
          'min_child_samples': 79,
          'objective': 'binary',
          'max_depth': 13,
          'learning_rate': 0.03,
          "boosting_type": "gbdt",
          "subsample_freq": 3,
          "subsample": 0.9,
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3,
          'reg_lambda': 0.3,
          'colsample_bytree': 0.9,
          #'categorical_feature': cat_cols
         }

adj_params = {'num_leaves': range(10, 300, 10),
              'max_depth': range(3, 16, 3),
              'min_child_weight': np.arange(0.001, 0.010, 0.001),
              'min_child_samples': np.arange(4, 100, 5),
              'subsample': [round(i,1) for i in np.arange(0.4,1.1,0.2)],
              'subsample_freq': range(0,6,1),
              'colsample_bytree': [round(i,1) for i in np.arange(0.4,1.1,0.2)],
              'reg_alpha': [round(i,2) for i in np.arange(0.0,0.1,0.01)],
              'reg_lambda': [round(i,2) for i in np.arange(0.0,0.1,0.01)],
              'learning_rate': [0.01,0.05,0.1,0.5]
             }

lgbc = LGBMClassifier(**params)
cscv = RandomSearch(lgbc , adj_params ,train_X, train_y, 5)