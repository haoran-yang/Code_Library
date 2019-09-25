# 超参数调优示例
```python
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from bayes_opt import BayesianOptimization
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, auc,precision_recall_curve
from sklearn import metrics
from sklearn import preprocessing
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import datasets
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from collections import Counter
import warnings
warnings.filterwarnings("ignore")
```

```python
#加载数据集
dataset = datasets.load_breast_cancer()
```

```python
X = dataset['data']
y = dataset['target']
```

```python
X.shape, Counter(y)
```
    ((569, 30), Counter({0: 212, 1: 357}))

```python
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=2)
```

```python
X_train.shape, X_test.shape, Counter(y_train), Counter(y_test)
```




    ((455, 30), (114, 30), Counter({1: 288, 0: 167}), Counter({1: 69, 0: 45}))




```python
# 使用默认参数
params = {'num_leaves':31, 'max_depth':-1, 'learning_rate':0.1, 'n_estimators':100, 'subsample_for_bin':200000, 'objective':'binary', 
         'min_split_gain':0.0, 'min_child_weight':0.001, 'min_child_samples':20, 'subsample':1.0, 'subsample_freq':0, 'colsample_bytree':1.0, 
         'reg_alpha':0.0, 'reg_lambda':0.0, 'random_state':6, 'metric':'auc'}
lgbc = lgb.LGBMClassifier(**params)
```

## 网格搜索（Grid Search）

### 全局搜索

- 对全部需要调节的参数设置范围和步长，对所有的参数组合进行搜索，找出最优

```python
def GridSearch(clf, params, X, y):
    cscv = GridSearchCV(clf, params, scoring='roc_auc', n_jobs=-1, cv=5)
    cscv.fit(X, y)
    return cscv
```

```python
%%time
adj_params = {'num_leaves': range(6, 100, 5),
              'max_depth': range(3, 15, 3),
              'min_child_weight': np.arange(0.001, 0.010, 0.001),
              'min_child_samples': np.arange(4, 30, 2),
              'subsample': [round(i,1) for i in np.arange(0.4,1.1,0.2)],
              'subsample_freq': range(0,6,1),
              'colsample_bytree': [round(i,1) for i in np.arange(0.4,1.1,0.2)],
              'reg_alpha': [round(i,2) for i in np.arange(0.0,0.1,0.01)],
              'reg_lambda': [round(i,2) for i in np.arange(0.0,0.1,0.01)]
             }
cscv = GridSearch(lgbc , adj_params , X_train, y_train)
```
### 手动搜索

```python
train_set = lgb.Dataset( X_train, y_train)

cv_result =lgb.cv(params=params,train_set=train_set,num_boost_round=1000,nfold=5,metrics='auc',early_stopping_rounds=50)

len(cv_result['auc-mean']), cv_result['auc-mean'][-1]
```
    (28, 0.9929415323298103)

```python
params.update({'n_estimators':28})
lgbc = lgb.LGBMClassifier(**params)
```

```python
%%time
adj_params = {'num_leaves': range(6, 100, 5),
             'max_depth': range(3, 15, 3)
             }
cscv = GridSearch(lgbc , adj_params , X_train, y_train)
print(cscv.best_score_, cscv.best_params_)
```
    0.9922602147861035 {'max_depth': 6, 'num_leaves': 11}
    Wall time: 2.75 s

```python
%%time
adj_params = {'num_leaves': [8,11,14],
             'max_depth': [5, 6, 7]
             }
cscv = GridSearch(lgbc , adj_params , X_train, y_train)
print(cscv.best_score_, cscv.best_params_)
```
    0.9922602147861035 {'max_depth': 6, 'num_leaves': 11}
    Wall time: 388 ms
    
```python
params.update({'max_depth': 6, 'num_leaves': 11})
lgbc = lgb.LGBMClassifier(**params)
```

```python
%%time
adj_params = {'min_child_weight': np.arange(0.001, 0.010, 0.001),
             'min_child_samples': np.arange(4, 30, 2)
             }
cscv = GridSearch(lgbc , adj_params , X_train, y_train)
print(cscv.best_score_, cscv.best_params_)
```
    0.9932855941610078 {'min_child_samples': 16, 'min_child_weight': 0.001}
    Wall time: 3.77 s
    
```python
params.update({'min_child_weight': 0.001, 'min_child_samples': 16})
lgbc = lgb.LGBMClassifier(**params)
```

```python
%%time
adj_params = {'subsample': [round(i,1) for i in np.arange(0.4,1.1,0.2)],
             'subsample_freq': range(0,6,1),
             'colsample_bytree': [round(i,1) for i in np.arange(0.4,1.1,0.2)]
             }
cscv = GridSearch(lgbc , adj_params , X_train, y_train)
print(cscv.best_score_, cscv.best_params_)
```
    0.9934524960467098 {'colsample_bytree': 0.4, 'subsample': 0.4, 'subsample_freq': 0}
    Wall time: 2.31 s
    
```python
params.update({'colsample_bytree': 0.4, 'subsample': 0.4, 'subsample_freq': 0})
lgbc = lgb.LGBMClassifier(**params)
```

```python
%%time
adj_params = {'reg_alpha': [round(i,2) for i in np.arange(0.0,0.1,0.01)],
              'reg_lambda': [round(i,2) for i in np.arange(0.0,0.1,0.01)]
             }
cscv = GridSearch(lgbc , adj_params , X_train, y_train)
print(cscv.best_score_, cscv.best_params_)
```

    0.9938586803751743 {'reg_alpha': 0.04, 'reg_lambda': 0.07}
    Wall time: 1.69 s
    
```python
params.update({'reg_alpha': 0.04, 'reg_lambda': 0.07})
```

```python
print(params)
```

    {'num_leaves': 11, 'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 28, 'subsample_for_bin': 200000, 'objective': 'binary', 'min_split_gain': 0.0, 'min_child_weight': 0.001, 'min_child_samples': 16, 'subsample': 0.4, 'subsample_freq': 0, 'colsample_bytree': 0.4, 'reg_alpha': 0.04, 'reg_lambda': 0.07, 'random_state': 6, 'metric': 'auc'}
    
```python
lgbc = lgb.LGBMClassifier(**params)
```

```python
cv_score = cross_val_score(estimator=lgbc,X=X,y=y,scoring='roc_auc',cv=5,n_jobs=-1).mean()
print(cv_score)
```
    0.9915777611404863
    
```python
lgbc.fit(X_train,y_train)

probs = lgbc.predict_proba(X_test)[:,1]

test_score = roc_auc_score(y_true=y_test,y_score=probs)
print(test_score)
```
    0.9826086956521739
    

## 随机搜索（Randomized Search）


```python
params = {'num_leaves':31, 'max_depth':-1, 'learning_rate':0.1, 'n_estimators':100, 'subsample_for_bin':200000, 'objective':'binary', 
         'min_split_gain':0.0, 'min_child_weight':0.001, 'min_child_samples':20, 'subsample':1.0, 'subsample_freq':0, 'colsample_bytree':1.0, 
         'reg_alpha':0.0, 'reg_lambda':0.0, 'random_state':6, 'metric':'auc'}
```

```python
def RandomSearch(clf, params, X, y, n_iter):
    cscv = RandomizedSearchCV(clf, params, n_iter=n_iter, scoring='roc_auc', n_jobs=-1, cv=5)
    cscv.fit(X, y)
    return cscv
```

```python
params.update({'n_estimators':28})
lgbc = lgb.LGBMClassifier(**params)
```

```python
%%time
adj_params = {'num_leaves': range(6, 100, 5),
              'max_depth': range(3, 15, 3),
              'min_child_weight': np.arange(0.001, 0.010, 0.001),
              'min_child_samples': np.arange(4, 30, 2),
              'subsample': [round(i,1) for i in np.arange(0.4,1.1,0.2)],
              'subsample_freq': range(0,6,1),
              'colsample_bytree': [round(i,1) for i in np.arange(0.4,1.1,0.2)],
              'reg_alpha': [round(i,2) for i in np.arange(0.0,0.1,0.01)],
              'reg_lambda': [round(i,2) for i in np.arange(0.0,0.1,0.01)]
             }
cscv = RandomSearch(lgbc , adj_params , X_train, y_train, 1000)
```
    Wall time: 22.5 s
    
```python
print(cscv.best_params_, cscv.best_score_)
```
    {'subsample_freq': 3, 'subsample': 0.8, 'reg_lambda': 0.03, 'reg_alpha': 0.07, 'num_leaves': 36, 'min_child_weight': 0.002, 'min_child_samples': 8, 'max_depth': 9, 'colsample_bytree': 0.6} 0.9943397800022118
    
```python
params.update(cscv.best_params_)
```

```python
print(params)
```
    {'num_leaves': 36, 'max_depth': 9, 'learning_rate': 0.1, 'n_estimators': 28, 'subsample_for_bin': 200000, 'objective': 'binary', 'min_split_gain': 0.0, 'min_child_weight': 0.002, 'min_child_samples': 8, 'subsample': 0.8, 'subsample_freq': 3, 'colsample_bytree': 0.6, 'reg_alpha': 0.07, 'reg_lambda': 0.03, 'random_state': 6, 'metric': 'auc'}
    
```python
lgbc = lgb.LGBMClassifier(**params)
```

```python
cv_score = cross_val_score(estimator=lgbc,X=X,y=y,scoring='roc_auc',cv=5,n_jobs=-1).mean()
print(cv_score)
```
    0.9916970817150969
    
```python
lgbc.fit(X_train,y_train)
probs = lgbc.predict_proba(X_test)[:,1]
roc_auc_score(y_true=y_test,y_score=probs)
```
    0.9864734299516907

## 贝叶斯优化（Bayesian Optimization）


```python
def BayesianSearch(clf, params):
    """贝叶斯优化器"""
    # 迭代次数
    num_iter = 25
    init_points = 5
    # 创建一个贝叶斯优化对象，输入为自定义的模型评估函数与超参数的范围
    bayes = BayesianOptimization(clf, params)
    # 开始优化
    bayes.maximize(init_points=init_points, n_iter=num_iter)
    return bayes
```

```python
def GBM_evaluate(num_leaves, max_depth, min_child_weight, min_child_samples, subsample, subsample_freq, colsample_bytree, reg_alpha, reg_lambda):
    """自定义的模型评估函数"""

    # 模型固定的超参数
    param = {
        'objective': 'binary',
        'n_estimators': 28,
        'metric': 'auc',
        'learning_rate':0.1,
        'random_state': 6}

    # 贝叶斯优化器生成的超参数
    param['min_child_weight'] = int(num_leaves)
    param['max_depth'] = int(max_depth)
    param['min_child_weight'] = float(min_child_weight)
    param['min_child_samples'] = int(min_child_samples)
    param['subsample'] = float(subsample)
    param['subsample_freq'] = int(subsample_freq)
    param['colsample_bytree'] = float(colsample_bytree)
    param['reg_lambda'] = float(reg_lambda)
    param['reg_alpha'] = float(reg_alpha)
    
    # 5-flod 交叉检验，注意BayesianOptimization会向最大评估值的方向优化，因此对于回归任务需要取负数。
    val = cross_val_score(lgb.LGBMClassifier(**param),X_train, y_train ,scoring='roc_auc', cv=5).mean()
    return val
```

```python
%%time
# 调参范围
adj_params = {'num_leaves': (6, 100),
              'max_depth': (3, 15),
              'min_child_weight': (0.001, 0.01),
              'min_child_samples': (4, 30),
              'subsample': (0.4, 1.0),
              'subsample_freq': (0, 6),
              'colsample_bytree': (0.4, 1.0),
              'reg_alpha': (0.0, 0.1),
              'reg_lambda': (0.0, 0.1)
             }
# 调用贝叶斯优化
bayesian_result = BayesianSearch(GBM_evaluate, adj_params)
```

    |   iter    |  target   | colsam... | max_depth | min_ch... | min_ch... | num_le... | reg_alpha | reg_la... | subsample | subsam... |
    -------------------------------------------------------------------------------------------------------------------------------------
    | [0m 1       [0m | [0m 0.992   [0m | [0m 0.6131  [0m | [0m 6.529   [0m | [0m 21.09   [0m | [0m 0.008365[0m | [0m 12.43   [0m | [0m 0.07952 [0m | [0m 0.03076 [0m | [0m 0.8642  [0m | [0m 2.161   [0m |
    | [95m 2       [0m | [95m 0.992   [0m | [95m 0.6005  [0m | [95m 14.37   [0m | [95m 21.94   [0m | [95m 0.001349[0m | [95m 59.36   [0m | [95m 0.01041 [0m | [95m 0.08223 [0m | [95m 0.8218  [0m | [95m 3.593   [0m |
    | [95m 3       [0m | [95m 0.9922  [0m | [95m 0.7575  [0m | [95m 7.302   [0m | [95m 23.36   [0m | [95m 0.009218[0m | [95m 52.97   [0m | [95m 0.08787 [0m | [95m 0.0914  [0m | [95m 0.8791  [0m | [95m 3.026   [0m |
    | [0m 4       [0m | [0m 0.989   [0m | [0m 0.5944  [0m | [0m 3.12    [0m | [0m 5.041   [0m | [0m 0.004724[0m | [0m 12.66   [0m | [0m 0.09395 [0m | [0m 0.09204 [0m | [0m 0.4741  [0m | [0m 0.537   [0m |
    | [0m 5       [0m | [0m 0.9914  [0m | [0m 0.861   [0m | [0m 8.03    [0m | [0m 22.39   [0m | [0m 0.001921[0m | [0m 60.9    [0m | [0m 0.07612 [0m | [0m 0.0383  [0m | [0m 0.8636  [0m | [0m 5.986   [0m |
    | [0m 6       [0m | [0m 0.9898  [0m | [0m 0.9407  [0m | [0m 4.135   [0m | [0m 29.81   [0m | [0m 0.002876[0m | [0m 99.97   [0m | [0m 0.05714 [0m | [0m 0.02759 [0m | [0m 0.5534  [0m | [0m 1.71    [0m |
    | [0m 7       [0m | [0m 0.9903  [0m | [0m 0.695   [0m | [0m 14.94   [0m | [0m 5.531   [0m | [0m 0.00903 [0m | [0m 99.74   [0m | [0m 0.01618 [0m | [0m 0.07275 [0m | [0m 0.8278  [0m | [0m 5.66    [0m |
    | [0m 8       [0m | [0m 0.9874  [0m | [0m 0.9468  [0m | [0m 13.0    [0m | [0m 29.63   [0m | [0m 0.005387[0m | [0m 6.173   [0m | [0m 0.07618 [0m | [0m 0.08057 [0m | [0m 0.5694  [0m | [0m 5.388   [0m |
    | [95m 9       [0m | [95m 0.9923  [0m | [95m 0.6939  [0m | [95m 3.535   [0m | [95m 4.161   [0m | [95m 0.008265[0m | [95m 99.78   [0m | [95m 0.06751 [0m | [95m 0.07202 [0m | [95m 0.5371  [0m | [95m 1.768   [0m |
    | [0m 10      [0m | [0m 0.992   [0m | [0m 0.7932  [0m | [0m 4.247   [0m | [0m 4.003   [0m | [0m 0.003177[0m | [0m 99.99   [0m | [0m 0.009365[0m | [0m 0.001658[0m | [0m 0.985   [0m | [0m 4.743   [0m |
    | [0m 11      [0m | [0m 0.9904  [0m | [0m 0.8216  [0m | [0m 3.167   [0m | [0m 5.114   [0m | [0m 0.006548[0m | [0m 99.97   [0m | [0m 0.07904 [0m | [0m 0.06183 [0m | [0m 0.8867  [0m | [0m 0.8045  [0m |
    | [0m 12      [0m | [0m 0.9902  [0m | [0m 0.6978  [0m | [0m 13.81   [0m | [0m 29.7    [0m | [0m 0.003554[0m | [0m 99.97   [0m | [0m 0.07383 [0m | [0m 0.04245 [0m | [0m 0.5456  [0m | [0m 1.39    [0m |
    | [0m 13      [0m | [0m 0.9895  [0m | [0m 0.6704  [0m | [0m 11.36   [0m | [0m 4.083   [0m | [0m 0.003091[0m | [0m 99.99   [0m | [0m 0.05901 [0m | [0m 0.03642 [0m | [0m 0.4346  [0m | [0m 0.1465  [0m |
    | [0m 14      [0m | [0m 0.9887  [0m | [0m 0.4298  [0m | [0m 3.682   [0m | [0m 29.85   [0m | [0m 0.003705[0m | [0m 99.98   [0m | [0m 0.08792 [0m | [0m 0.02495 [0m | [0m 0.4078  [0m | [0m 4.892   [0m |
    | [0m 15      [0m | [0m 0.987   [0m | [0m 0.9691  [0m | [0m 3.202   [0m | [0m 4.217   [0m | [0m 0.001766[0m | [0m 6.145   [0m | [0m 0.0282  [0m | [0m 0.01572 [0m | [0m 0.4677  [0m | [0m 4.118   [0m |
    | [0m 16      [0m | [0m 0.9873  [0m | [0m 0.5977  [0m | [0m 14.67   [0m | [0m 29.97   [0m | [0m 0.005972[0m | [0m 99.97   [0m | [0m 0.00283 [0m | [0m 0.05836 [0m | [0m 0.4112  [0m | [0m 5.834   [0m |
    | [0m 17      [0m | [0m 0.9911  [0m | [0m 0.5516  [0m | [0m 3.56    [0m | [0m 29.85   [0m | [0m 0.00607 [0m | [0m 49.5    [0m | [0m 0.07928 [0m | [0m 0.03027 [0m | [0m 0.9918  [0m | [0m 0.4168  [0m |
    | [0m 18      [0m | [0m 0.9912  [0m | [0m 0.6254  [0m | [0m 14.9    [0m | [0m 5.009   [0m | [0m 0.001182[0m | [0m 58.52   [0m | [0m 0.07352 [0m | [0m 0.0144  [0m | [0m 0.9123  [0m | [0m 0.06782 [0m |
    | [0m 19      [0m | [0m 0.9874  [0m | [0m 0.6099  [0m | [0m 3.327   [0m | [0m 4.973   [0m | [0m 0.003761[0m | [0m 65.57   [0m | [0m 0.08579 [0m | [0m 0.07034 [0m | [0m 0.97    [0m | [0m 0.2418  [0m |
    | [95m 20      [0m | [95m 0.9924  [0m | [95m 0.4089  [0m | [95m 14.77   [0m | [95m 4.48    [0m | [95m 0.004031[0m | [95m 99.98   [0m | [95m 0.03215 [0m | [95m 0.08321 [0m | [95m 0.7844  [0m | [95m 2.771   [0m |
    | [0m 21      [0m | [0m 0.9911  [0m | [0m 0.8009  [0m | [0m 14.9    [0m | [0m 4.304   [0m | [0m 0.004796[0m | [0m 99.84   [0m | [0m 0.0431  [0m | [0m 0.08972 [0m | [0m 0.906   [0m | [0m 0.01622 [0m |
    | [0m 22      [0m | [0m 0.9881  [0m | [0m 0.8644  [0m | [0m 14.71   [0m | [0m 4.073   [0m | [0m 0.008128[0m | [0m 99.58   [0m | [0m 0.08806 [0m | [0m 0.01648 [0m | [0m 0.4147  [0m | [0m 4.903   [0m |
    | [0m 23      [0m | [0m 0.9909  [0m | [0m 0.777   [0m | [0m 3.612   [0m | [0m 29.78   [0m | [0m 0.001354[0m | [0m 99.76   [0m | [0m 0.04696 [0m | [0m 0.001682[0m | [0m 0.567   [0m | [0m 0.2422  [0m |
    | [0m 24      [0m | [0m 0.9905  [0m | [0m 0.5621  [0m | [0m 3.019   [0m | [0m 29.87   [0m | [0m 0.004477[0m | [0m 99.5    [0m | [0m 0.00124 [0m | [0m 0.005836[0m | [0m 0.4111  [0m | [0m 0.4989  [0m |
    | [0m 25      [0m | [0m 0.9907  [0m | [0m 0.9562  [0m | [0m 3.888   [0m | [0m 4.85    [0m | [0m 0.009633[0m | [0m 99.88   [0m | [0m 0.05619 [0m | [0m 0.06429 [0m | [0m 0.7565  [0m | [0m 0.687   [0m |
    | [0m 26      [0m | [0m 0.9911  [0m | [0m 0.7468  [0m | [0m 13.78   [0m | [0m 29.51   [0m | [0m 0.00755 [0m | [0m 99.99   [0m | [0m 0.04964 [0m | [0m 0.01591 [0m | [0m 0.4915  [0m | [0m 0.2551  [0m |
    | [0m 27      [0m | [0m 0.9906  [0m | [0m 0.643   [0m | [0m 14.64   [0m | [0m 28.88   [0m | [0m 0.001416[0m | [0m 99.59   [0m | [0m 0.05849 [0m | [0m 0.008487[0m | [0m 0.7714  [0m | [0m 0.4789  [0m |
    | [0m 28      [0m | [0m 0.9917  [0m | [0m 0.8947  [0m | [0m 3.061   [0m | [0m 29.98   [0m | [0m 0.006696[0m | [0m 99.97   [0m | [0m 0.07466 [0m | [0m 0.03479 [0m | [0m 0.9447  [0m | [0m 1.3     [0m |
    | [0m 29      [0m | [0m 0.9911  [0m | [0m 0.7183  [0m | [0m 3.214   [0m | [0m 28.64   [0m | [0m 0.007787[0m | [0m 99.98   [0m | [0m 0.05544 [0m | [0m 0.01526 [0m | [0m 0.5139  [0m | [0m 1.253   [0m |
    | [0m 30      [0m | [0m 0.9894  [0m | [0m 0.5281  [0m | [0m 3.042   [0m | [0m 29.98   [0m | [0m 0.006397[0m | [0m 99.9    [0m | [0m 0.0059  [0m | [0m 0.03192 [0m | [0m 0.7829  [0m | [0m 1.01    [0m |
    =====================================================================================================================================
    Wall time: 5min 15s
    


```python
print('Best bayesian score: %s\n'%bayesian_result.max['target'])
print('Best bayesian params: %s'%str(bayesian_result.max['params']))
```
    Best bayesian score: 0.9924000871315913
    
    Best bayesian params: {'colsample_bytree': 0.4089206496876992, 'max_depth': 14.773819480466189, 'min_child_samples': 4.480196571648507, 'min_child_weight': 0.004031333374775868, 'num_leaves': 99.97968887651052, 'reg_alpha': 0.03214605103841293, 'reg_lambda': 0.08320635117193971, 'subsample': 0.7843722149752079, 'subsample_freq': 2.770536702654322}
    
```python
params.update(bayesian_result.max['params'])
```


```python
for p in ['num_leaves','max_depth','min_child_samples','subsample_freq']:
    params[p] = int(params.get(p))
```


```python
print(params)
```

    {'num_leaves': 99, 'max_depth': 14, 'learning_rate': 0.1, 'n_estimators': 28, 'subsample_for_bin': 200000, 'objective': 'binary', 'min_split_gain': 0.0, 'min_child_weight': 0.004031333374775868, 'min_child_samples': 4, 'subsample': 0.7843722149752079, 'subsample_freq': 2, 'colsample_bytree': 0.4089206496876992, 'reg_alpha': 0.03214605103841293, 'reg_lambda': 0.08320635117193971, 'random_state': 6, 'metric': 'auc'}
    


```python
lgbc = lgb.LGBMClassifier(**params)
```


```python
cv_score = cross_val_score(estimator=lgbc,X=X,y=y,scoring='roc_auc',cv=5,n_jobs=-1).mean()
print(cv_score)
```

    0.9910362692953587
    


```python
lgbc.fit(X_train,y_train)
probs = lgbc.predict_proba(X_test)[:,1]
roc_auc_score(y_true=y_test,y_score=probs)
```




    0.9832528180354266


