import pandas as pd
import numpy as np
import time, datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import math
from collections import Counter
import xgboost
import lightgbm
from sklearn import ensemble
from pyecharts import Bar,Line,Overlap,Grid,Page
from scipy import stats

def resumetable(df):
    '''了解数据'''
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values
    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 
    return summary

def reduce_mem_usage(df):
    '''降低数据对内存占用'''
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# 找出唯一取值列
def one_unique_cols(data,cols=[]):
    '''筛选只有唯一取值的特征列，cols默认为data所有列'''
    oneUnCol = []
    cols = data.columns if not cols else cols
    for col in cols:
        if data[col].nunique(dropna=False)==1:
            oneUnCol.append(col)
    return oneUnCol
# 找出分布极端列
def extreme_imbalance_cols(data,cols=[],threshold=0.99):
    '''筛选取值极端分布列，cols默认为data所有列，threshold默认0.99'''
    exImbCol = []
    cols = data.columns if not cols else cols
    for col in cols:
        if data[col].value_counts(dropna=False)[0]/data.shape[0] >= threshold:
            exImbCol.append(col)
    return exImbCol

def strNan_to_nan(data):
    '''str类型空值替换为np.nan'''
    return data.replace(['nan','NaN','',' ','NaT','NA','None',None],np.nan)

def null_ratio_counts(data,value=0.9):
    """
    缺失率统计
    缺失比例一般达到90%则可舍弃，也可以将值替换为有值(1)和空值(0)，继续作为特征
    param:
          data: dataframe
          value:阈值，返回缺失比例>=该值的特征名,默认0.9
    return:
          缺失统计表
          比例较高特征列表
    """
    data_null=data.isnull().sum().sort_values(ascending=False)
    data_null_percent=data_null/data.shape[0]
    null_data=pd.concat([data_null,data_null_percent],axis=1)
    null_data.rename(columns={0:'NullCounts',1:'NullRatio'},inplace=True)
    null_data_style=null_data[null_data['NullCounts']>0]
    return null_data_style.style.bar(), list(null_data[null_data['NullRatio']>=value].index)

def divide_x_dtype(data):
    """
    划分数据类型
    注：缺失值需为np.nan
    """
    numeric_X=[]
    category_X=[]
    encode_X = []
    for col,ser in data.iteritems():
        ser_t = ser[ser.notnull()].astype(str)
        try:
            if ser_t.nunique()<=5:
                category_X.append(col) # 取值数小于等于5划入类别特征
            else:
                ser_t.apply(float)
                try:
                    # 首字符为'0'，字符长度>1(排除'0'取值)，字符中不含'.'(排除'0.0')
                    encode_nums = ((ser_t.str[:1]=='0')&(ser_t.str.len()>1)&(ser_t.apply(lambda x:False if '.' in x else True))).sum()
                except:
                    print('code error!')
                if encode_nums>0:
                    category_X.append(col)
                    encode_X.append(col)
                    print('%s 编码取值量:%s, 取值率:%.2f'%(col,encode_nums,encode_nums/len(ser_t)))
                else:
                    numeric_X.append(col)
        except:
            category_X.append(col)
    print('-'*40)
    print('总计%s个含编码类别特征:%s'%(len(encode_X),encode_X))
    return numeric_X, category_X

def split_datetime(dt_ser,datetime_type=1):
    '''拆分日期时间。
    datetime_type:为"1"时，输入格式如"2019-01-18 21:03:46"
                  为"2"时，输入格式如"20190118" '''
    datimeElements = pd.DataFrame(index=dt_ser.index)
    if dt_ser.dtype in ['float32','float64','int32','int64']:
        print('The type must be object or datetime!')
    else:
        if dt_ser.dtype == 'object':
            if datetime_type==1:
                dt_ser = dt_ser.apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
            elif datetime_type==2:
                dt_ser = dt_ser.apply(lambda x:datetime.datetime.strptime(x,'%Y%m%d'))
        datimeElements['month'] = dt_ser.apply(lambda x:x.month)
        datimeElements['day'] = dt_ser.apply(lambda x:x.day)
        datimeElements['weekOfYear'] = dt_ser.apply(lambda x:x.week)
        datimeElements['weekOfMth'] = dt_ser.apply(lambda x:(x.day - 1) // 7 + 1)
        datimeElements['dayOfWeek'] = dt_ser.apply(lambda x:x.strftime('%w'))
        if datetime_type==1:
            datimeElements['hour'] = dt_ser.apply(lambda x:x.hour)
            datimeElements['minute'] = dt_ser.apply(lambda x:x.minute)
    return datimeElements


def vif_compute(data,index):
    '''方差膨胀因子(VIF)：大于10存在严重多重共线性
    data:frame 自变量
    index:被解释变量的索引位置'''
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    return variance_inflation_factor(data.values, index)

def feature_importance_ensemble(x_array,y_array,xName_list,n_estimators=100, max_depth=3,figsize=(22,8)):
    '''树模型特征重要性计算、排序和绘图'''
    xgbc = xgboost.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth)
    lgbc = lightgbm.LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth)
    rfc = ensemble.RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    xgbc.fit(X=x_array,y=y_array)
    lgbc.fit(X=x_array,y=y_array)
    rfc.fit(X=x_array,y=y_array)
    xgbc_imp = pd.Series(xgbc.feature_importances_,index=xName_list,name='importance_xgbc')
    lgbc_imp = pd.Series(lgbc.feature_importances_,index=xName_list,name='importance_lgbc')
    rfc_imp = pd.Series(rfc.feature_importances_,index=xName_list,name='importance_rfc')
    imp_df=pd.concat([xgbc_imp,lgbc_imp,rfc_imp],axis=1)
    for i in ['importance_xgbc','importance_lgbc','importance_rfc']:
         imp_df['rank_'+i.split('_')[1]] = imp_df[i].rank(ascending=False)
    imp_df['rank_Total'] = imp_df[['rank_xgbc','rank_lgbc','rank_rfc']].sum(axis=1).rank()
    imp_df = imp_df.sort_values('rank_Total')
    # 重要性绘图
    plt.figure(figsize=figsize)
    n=1
    for i in ['importance_xgbc','importance_lgbc','importance_rfc']:
        plt.subplot(1,3,n)
        g = sns.barplot(y=imp_df.index, x=imp_df[i].values)
        g.set_title(i)
        if n==1:
            g.set_ylabel('Total Importances Ranking')
        n+=1
    return imp_df