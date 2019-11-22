import lightgbm as lgb
import xgboost as xgb
from sklearn import metrics
import pandas as pd
import numpy as np

# xgboost cv函数
def xgb_fit(alg, dtrain, fets,target, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        # 构造矩阵数据
        xgtrain = xgb.DMatrix(dtrain[fets].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    alg.fit(dtrain[fets], dtrain[target],eval_metric='auc')
    dtrain_predictions = alg.predict(dtrain[fets])
    dtrain_predprob = alg.predict_proba(dtrain[fets])[:,1]

    print("Accuracy : %.4g")%metrics.accuracy_score(dtrain[target].values, dtrain_predictions)
    print("AUC Score (Train): %f")% metrics.roc_auc_score(dtrain[target], dtrain_predprob)
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')

# lightgbm cv示例：固定学习率，确定最佳提升器个数
def lgb_fit(lgb_init,train_X,train_y,early_stopping_rounds=50):
    data_train = lgb.Dataset(train_X, train_y, silent=True)
    cv_results = lgb.cv(lgb_init, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='auc',
                        early_stopping_rounds=early_stopping_rounds, verbose_eval=50, show_stdv=False, seed=0)
    print('best n_estimators:', len(cv_results['auc-mean']))
    print('best cv score:', cv_results['auc-mean'][-1])

# 1.xgboost
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

xgb_gdsh = {'max_depth':range(3,10,2), # 1.max_depth与min_child_weight一起调
            'min_child_weight':range(1,6,2),
            'gamma':[i/10.0 for i in range(0,5)], # 2.gamma单独调
            'subsample':[i/10.0 for i in range(6,10)], # 3.subsample和colsample_bytree一起调
            'colsample_bytree':[i/10.0 for i in range(6,10)],
            'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100], # 4.reg_alpha和reg_lambda一起调
            'reg_lambda':[1e-5, 1e-2, 0.1, 1, 100]
                }

# 2.lightgbm
lgb_init = {'learning_rate':0.1,
            'n_estimators':1000,
            'max_depth': 5,
            'num_leaves': 64,
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

lgb_gdsh = {  'max_depth': range(3,10,2), # 1.max_depth和num_leaves一起调节
            'num_leaves': range(50,170,30),
            'min_child_samples': [18, 19, 20, 21, 22], # 2.min_child_samples和min_child_weight一起调节
            'min_child_weight': [0.001, 0.002],
            'subsample': [0.5, 0.6, 0.7, 0.8, 0.9,1.0], # 3.subsample、subsample_freq、colsample_bytree一起调节
            'subsample_freq': [1, 5],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5], # 4.reg_alpha和reg_lambda一起调节
            'reg_lambda': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5]
            }

# 3.逻辑回归
# solver:'newton-cg', 'lbfgs', 'sag'只适用于L2惩罚项的优化，liblinear、saga两种（L1和L2）都适用
# 大多数情况下可以直接指定solver和penalty，只对C做K则交叉验证调参。
logit_init = {
    'penalty':'l2', # 默认l2正则化
    'solver':'sag', # 默认liblinear，小数据集较好。sag、saga适合大数据集。
    'C':1.0, #越小正则化越强
    'class_weight':'balanced', #默认None
    'random_state':2,
    'n_jobs':-1
    }
# L1 L2正则化选择
logit_gdsh1 = {
    'C': np.arange(0.01,2,0.01),
    'penalty':['l1','l2'],
    'solver':['saga','liblinear']
}
# L2正则化下solver选择
logit_gdsh2 = {
    'C': np.arange(0.01,2,0.01),
    'penalty':['l2'],
    'solver':['newton-cg', 'lbfgs', 'sag','liblinear','saga']
}

# 4.支持向量机
svc_init = {'C':1.0, 
            'kernel':'rbf', # 默认高斯核
            'gamma':'auto_deprecated', 
            'random_state':2}

svc_gdsh = {'C':[0.1,1.0,10.0], 
                'kernel':['rbf','poly','linear'], 
                'gamma':['auto',0.1,0.01,0.001], # 默认auto（1/n_features），
                'random_state':2}

# 5.随机森林
rf_init = {'n_estimators':100, 
            'max_depth':5, 
            'min_samples_split':20, 
            'min_samples_leaf':10, 
            'max_features':'auto', 
            'min_weight_fraction_leaf':0.0, # 最小叶节点样本权重和，默认不考虑权重
            'max_leaf_nodes':None,  # 最大叶节点数，默认不限制
            'bootstrap':True, 
            'oob_score':True, # 默认False,是否使用out-of-bag（袋外样本）评估准确性
            'n_jobs':-1, 
            'random_state':2, 
            'warm_start':True, # 默认False。热启动。 
            'class_weight':'balanced'}

rf_gdsh = {'n_estimators':range(50,400,20), # 1.
            'max_depth':range(3,14,2),  # 2.max_depth、min_samples_split一起调节
            'min_samples_split':range(10,100,20),  # 3.min_samples_split、min_samples_leaf一起调节
            'min_samples_leaf':range(5,50,10), 
            'max_features':['auto',0.5,0.7,0.9] # 4.默认sqrt
            }

# 6.GBDT
gbdt_init = {'learning_rate':0.1, 
            'n_estimators':80, 
            'max_depth':5, 
            'min_samples_split':20, 
            'min_samples_leaf':10,
            'max_features':0.8, 
            'subsample':0.8, 
            'random_state':2,
            'warm_start':False # 默认False
            }

gbdt_gdsh = {'learning_rate':np.arange(0.01,0.5,0.02), 
            'n_estimators':range(50,151,20),  #1.选择树的合适个数
            'max_depth':[3,5,7,9],  # 2.max_depth和min_samples_split一起调节
            'min_samples_split':range(20,111,20),  # 3.min_samples_split和min_samples_leaf一起调节
            'min_samples_leaf':range(10,61,10),
            'max_features':range(10,30,2), # 4.根据特征数调节范围，可以和subsample一起调节
            'subsample':[0.5,0.7,0.9], 
            'warm_start':[True] # 默认False
            }