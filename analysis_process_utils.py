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
            ser_t.apply(float)
            try:
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


class FeatureLabelSizeRatio():
    '''单个特征按label统计'''
    def __init__(self):
        pass

    def get_woe_iv(self,df,x,x_type,y,only_total_iv=False,style=True,sort=False):
        '''单个变量WOE和IV值计算。与get_count_ratio区别：计算woe和iv。return df 和 overlap。only_total_iv=True时只return总IV。'''
        df_concat=df[[x,y]]    
        df_concat[y] = df_concat[y].apply(float)
        x_values=list(df_concat[x].unique())
        if x_type == 'numeric':
            x_values.sort()
        s_total=df_concat.groupby(y).size()
        woe_dic={}
        iv_dic={}
        total_iv=0
        for i in range(len(x_values)):
            a=df_concat[df_concat[x]==x_values[i]].groupby(y).size()
            aa=a.to_dict()
            for k in [0,1]:
                if aa.get(k,'null') == 'null':
                    aa[k]=1
            a=pd.Series(aa)
            respand_rate=a[1]/s_total[1]
            norespand_rate=a[0]/s_total[0]
            rate=respand_rate/norespand_rate
            woe=math.log(rate,math.e)
            iv=(a[1]/s_total[1]-a[0]/s_total[0])*woe
            total_iv += iv
            woe_dic[x_values[i]]=woe
            iv_dic[x_values[i]]=iv
        df_woe_iv=pd.DataFrame({x+'_WOE':woe_dic,x+'_IV':iv_dic}).T
        x_pivot=df_concat.groupby(y)[x].apply(Counter)
        x_pivot=x_pivot.unstack().T
        x_pivot.fillna(0,inplace=True)
        x_pivot['样本总量']=x_pivot[0]+x_pivot[1]
        x_pivot['坏样本占比']=x_pivot[1]/x_pivot['样本总量']
        x_des_woe_iv=pd.concat([x_pivot,df_woe_iv.T],axis=1)
        if sort:
            x_des_woe_iv = x_des_woe_iv.sort_values(by='坏样本占比')
        if not only_total_iv:
            bar = Bar(x,title_pos='center')  # 绘图
            line = Line()
            x_len = len(x_des_woe_iv.index)
            rotate = 0 if x_len < 15 else 45 if x_len < 30 else 90
            bar.add('count', x_des_woe_iv.index, x_des_woe_iv['样本总量'].values, yaxis_min=0, yaxis_max=None, is_label_show=False, label_pos='inside',
                    legend_pos='right', legend_orient='vertical',is_stack=False,label_color=['#35b0ab'],xaxis_rotate=rotate)
            line.add('bad_ratio', x_des_woe_iv.index, x_des_woe_iv['坏样本占比'].values, yaxis_min=0, yaxis_max=None, is_label_show=False, label_pos='inside',
                    legend_pos='right', legend_orient='vertical',is_stack=False,label_color=['#f34573'],line_width=1.5,is_smooth=False)
            overlap = Overlap(width='100%')
            overlap.add(bar)
            overlap.add(line,is_add_yaxis=True,yaxis_index=1)
        if only_total_iv:
            return total_iv
        else:
            print('total IV of {} is: {}'.format(x,total_iv))
            if style:
                return x_des_woe_iv.style.bar(subset=['样本总量','坏样本占比',x+'_IV',x+'_WOE']), overlap
            else:
                return x_des_woe_iv, overlap

    def get_sampleSize_oneRatio(self,data,x,xtype,y,value,style=True):
        '''单个特征-Y值分布统计。与get_woe_iv区别：不计算woe和iv。y为float型，取值0和1。x为float(数值型)或object(类别型)。xtype:"numeric"或其他。value:统计值，如"LOAN_NO"
        return style形式dataframe 和 overlap'''
        counts_ratio=pd.pivot_table(data=data[[x,y,value]],values=value,index=x,columns=y,aggfunc=len,margins=True,fill_value=0)
        counts_ratio['one_ratio'] = counts_ratio[1].div(counts_ratio['All'])
        counts_ratio = counts_ratio.drop(counts_ratio.index[-1])
        if xtype != 'numeric':
            counts_ratio = counts_ratio.sort_values('one_ratio')
        bar = Bar(x,title_pos='center')
        line = Line()
        x_len = len(counts_ratio.index)
        rotate = 0 if x_len < 15 else 45 if x_len < 30 else 90
        bar.add('count', counts_ratio.index, counts_ratio['All'].values, yaxis_min=0, yaxis_max=None, is_label_show=False, label_pos='inside',
                legend_pos='right', legend_orient='vertical',is_stack=False,label_color=['#35b0ab'],xaxis_rotate=rotate)
        line.add('y=1', counts_ratio.index, counts_ratio['one_ratio'].values, yaxis_min=0, yaxis_max=None, is_label_show=False, label_pos='inside',
                legend_pos='right', legend_orient='vertical',is_stack=False,label_color=['#f34573'],line_width=1.5,is_smooth=False)
        overlap = Overlap(width='100%')
        overlap.add(bar)
        overlap.add(line,is_add_yaxis=True,yaxis_index=1)
        if style:
            return counts_ratio.style.bar(subset=['All','one_ratio']), overlap
        else:
            return counts_ratio, overlap


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