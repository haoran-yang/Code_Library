import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import gc
from functools import partial
import time
from sklearn.metrics import make_scorer, roc_auc_score
## Hyperopt modules
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING


# 1. 随机网格搜索
def RandomSearch(clf, params, X, y, n_iter):
    cscv = RandomizedSearchCV(clf, params, n_iter=n_iter, scoring='roc_auc', n_jobs=-1, cv=5)
    cscv.fit(X, y)
    return cscv

# 2. Bayesian Optimization
def objective_xgb(params):
    '''xgb贝叶斯优化'''
    time1 = time.time()
    print("\n############## New Run ################")
    print(f"params = {params}")
    FOLDS = 5
    count=1
    param = {
            'learning_rate': "{:.3f}".format(params['learning_rate']),
            'n_estimators':int(params['n_estimators']),
            'max_depth': int(params['max_depth']),
            'min_child_weight': '{:.3f}'.format(params['min_child_weight']),
            'gamma': "{:.3f}".format(params['gamma']),
            'subsample': "{:.1f}".format(params['subsample']),
            'colsample_bytree': '{:.1f}'.format(params['colsample_bytree']),
            'reg_alpha': "{:.3f}".format(params['reg_alpha']),
            'reg_lambda': "{:.3f}".format(params['reg_lambda']),    
            }
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    # tss = TimeSeriesSplit(n_splits=FOLDS)
    # y_preds = np.zeros(test.shape[0])
    # y_oof = np.zeros(train_X.shape[0])
    score_mean = 0
    for tr_idx, val_idx in skf.split(train_X, train_y): #tss.split(train_X, train_y)
        clf = xgb.XGBClassifier( random_state=4, 
                            #   tree_method='gpu_hist', 
                                **param)
        X_tr, X_vl = train_X.iloc[tr_idx, :], train_X.iloc[val_idx, :]
        y_tr, y_vl = train_y.iloc[tr_idx], train_y.iloc[val_idx]
        clf.fit(X_tr, y_tr)
        score = make_scorer(roc_auc_score, needs_proba=True)(clf, X_vl, y_vl)
        score_mean += score
        print(f'{count} CV - score: {round(score, 4)}')
        count += 1
    time2 = time.time() - time1
    print(f"Total Time Run: {round(time2 / 60,2)}")
    gc.collect()
    print(f'Mean ROC_AUC: {score_mean / FOLDS}')
    del X_tr, X_vl, y_tr, y_vl, clf, score
    return -(score_mean / FOLDS)

def objective_lgb(params):
    """lightgbm贝叶斯优化"""
    time1 = time.time()
    print("\n############## New Run ################")
    print(f"params = {params}")
    FOLDS = 5
    count=1
    param = {
    'learning_rate':'{:.2f}'.format(params['learning_rate']),
    'n_estimators':int(params['n_estimators']),
    'max_depth': int(params['max_depth']),
    'num_leaves': int(params['num_leaves']),
    'min_child_samples': int(params['min_child_samples']),
    'min_child_weight': '{:.3f}'.format(params['min_child_weight']),
    'subsample': "{:.2f}".format(params['subsample']),
    'subsample_freq':int(params['subsample_freq']),
    'colsample_bytree': '{:.2f}'.format(params['colsample_bytree']),
    'reg_alpha': "{:.3f}".format(params['reg_alpha']),
    'reg_lambda': "{:.3f}".format(params['reg_lambda'])
    }
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
#     tss = TimeSeriesSplit(n_splits=FOLDS)
#     y_preds = np.zeros(test.shape[0])
    # y_oof = np.zeros(train_X.shape[0])
    score_mean = 0
    for tr_idx, val_idx in skf.split(train_X, train_y): #tss.split(train_X, train_y)
        clf = lgb.LGBMClassifier(random_state=4, **param)
        X_tr, X_vl = train_X.iloc[tr_idx, :], train_X.iloc[val_idx, :]
        y_tr, y_vl = train_y.iloc[tr_idx], train_y.iloc[val_idx]
        clf.fit(X_tr, y_tr)
        #y_pred_train = clf.predict_proba(X_vl)[:,1]
        #print(y_pred_train)
        score = make_scorer(roc_auc_score, needs_proba=True)(clf, X_vl, y_vl)
        # plt.show()
        score_mean += score
        print(f'{count} CV - score: {round(score, 4)}')
        count += 1
    time2 = time.time() - time1
    print(f"Total Time Run: {round(time2 / 60,2)}")
    gc.collect()
    print(f'Mean ROC_AUC: {score_mean / FOLDS}')
    del X_tr, X_vl, y_tr, y_vl, clf, score
    return -(score_mean / FOLDS)

if __name__ == "__main__":
    train = pd.read_csv('train.csv')
    train_y = train['target'].as_matrix()
    train_X = train.drop(columns='target').as_matrix()

    # initialized params of lgb/xgb
    lgb_init = {'learning_rate':0.1,
            'n_estimators':525,
            'max_depth': 9,
            'num_leaves': 128,
            'min_child_samples': 20,
            'min_child_weight':0.001,
            'min_split_gain':0.0,
            'subsample': 0.8,
            'subsample_freq':1,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state':2,
            'importance_type':'gain', # 默认'split'
            'objective': 'binary',
            'metric': 'auc',
           }
    xgb_init={'learning_rate':0.1,
          'n_estimators':800,
          'max_depth':5,
          'min_child_weight':1,
          'gamma':0,
          'subsample':0.8,
          'colsample_bytree':0.8,
          'reg_alpha':0,
          'reg_lambda':1,
          'objective':'binary:logistic',
          'scale_pos_weight':1,
          'seed':2,
          'random_state':0,
          'n_jobs':1}

    # 1. for random search 
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

    lgbc = lgb.LGBMClassifier(**params)
    cscv = RandomSearch(lgbc , adj_params ,train_X, train_y, 5)

    # 1. lightgbm bayesian
    lgb_space = {
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
        'n_estimators':hp.quniform('n_estimators', 100, 800, 10),
        'max_depth': hp.quniform('max_depth', 3, 23, 1),
        'num_leaves': hp.choice('num_leaves', list(range(20, 250, 10))),
        'min_child_samples': hp.choice('min_child_samples', list(range(100, 250, 10))),
        'min_child_weight': hp.choice('min_child_weight', list(np.arange(0.0001,0.1,0.0001))),
        'subsample': hp.choice('subsample', [0.2, 0.4, 0.5, 0.6, 0.7, .8, .9]),
        'subsample_freq': hp.quniform('subsample_freq', 1, 10, 1),
        'colsample_bytree': hp.choice('colsample_bytree', [0.2, 0.4, 0.5, 0.6, 0.7, .8, .9]),
        'reg_alpha':  hp.uniform('reg_alpha', 0.01, 0.4),
        'reg_lambda': hp.uniform('reg_lambda', 0.01, 0.4),
    }
    # Set algoritm parameters
    best_lgb = fmin(fn=objective_lgb,
                space=lgb_space,
                algo=tpe.suggest,
                max_evals=10)
    # Print best parameters
    best_params_lgb = space_eval(lgb_space, best_lgb)

    # 2. xgboost Bayesian
    xgb_space = {
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
        'n_estimators':hp.quniform('n_estimators', 100, 800, 10),
        'max_depth': hp.quniform('max_depth', 3, 23, 1),
        'min_child_weight': hp.choice('min_child_weight', list(np.arange(0.0001,0.1,0.0001))),
        'gamma': hp.uniform('gamma', 0.001, 0.7),
        'subsample': hp.choice('subsample', [0.2, 0.4, 0.5, 0.6, 0.7, .8, .9]),
        'colsample_bytree': hp.choice('colsample_bytree', [0.2, 0.4, 0.5, 0.6, 0.7, .8, .9]),
        'reg_alpha':  hp.uniform('reg_alpha', 0.01, 0.4),
        'reg_lambda': hp.uniform('reg_lambda', 0.01, 0.4),
        }
    best_xgb = fmin(fn=objective_xgb,
                space=xgb_space,
                algo=tpe.suggest,
                max_evals=5)
    # Print best parameters
    best_params_xgb = space_eval(xgb_space, best_xgb)