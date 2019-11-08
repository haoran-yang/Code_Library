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
def objective(params):
    '''贝叶斯优化'''
    time1 = time.time()
    print("\n############## New Run ################")
    print(f"params = {params}")
    FOLDS = 5
    count=1
    param = {
            'max_depth': int(params['max_depth']),
            'gamma': "{:.3f}".format(params['gamma']),
            'subsample': "{:.2f}".format(params['subsample']),
            'reg_alpha': "{:.3f}".format(params['reg_alpha']),
            'reg_lambda': "{:.3f}".format(params['reg_lambda']),
            'learning_rate': "{:.3f}".format(params['learning_rate']),
            # 'num_leaves': '{:.3f}'.format(params['num_leaves']),
            'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
            # 'min_child_samples': '{:.3f}'.format(params['min_child_samples']),
            'feature_fraction': '{:.3f}'.format(params['feature_fraction']),
            'bagging_fraction': '{:.3f}'.format(params['bagging_fraction']),
            #  'min_child_samples':'{:.3f}'.format(params['min_child_samples']),
            #  'subsample_freq':'{:.3f}'.format(params['subsample_freq']),
            #  'min_child_weight': '{:.3f}'.format(params['subsample_freq'])
            }
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    # tss = TimeSeriesSplit(n_splits=FOLDS)
    # y_preds = np.zeros(test.shape[0])
    # y_oof = np.zeros(train_X.shape[0])
    score_mean = 0
    for tr_idx, val_idx in skf.split(train_X, train_y): #tss.split(train_X, train_y)
        clf = lgb.LGBMClassifier(n_estimators=300, random_state=4, 
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

if __name__ == "__main__":
    train = pd.read_csv('train.csv')
    train_y = train['target'].as_matrix()
    train_X = train.drop(columns='target').as_matrix()

    # initialized params of lgb
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

    # 2. for Bayesian
    space = {
            'max_depth': hp.quniform('max_depth', 7, 23, 1),
            'reg_alpha':  hp.uniform('reg_alpha', 0.01, 0.4),
            'reg_lambda': hp.uniform('reg_lambda', 0.01, .4),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.3, .9),
            'gamma': hp.uniform('gamma', 0.01, .7),
            'num_leaves': hp.choice('num_leaves', list(range(20, 250, 10))),
            'min_child_samples': hp.choice('min_child_samples', list(range(100, 250, 10))),
            'subsample': hp.choice('subsample', [0.2, 0.4, 0.5, 0.6, 0.7, .8, .9]),
            'feature_fraction': hp.uniform('feature_fraction', 0.4, .8),
            'bagging_fraction': hp.uniform('bagging_fraction', 0.4, .9)
            }
    # Set algoritm parameters
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=10)
    # Print best parameters
    best_params = space_eval(space, best)