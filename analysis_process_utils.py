import pandas as pd
import numpy as np
import time, datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import math
from collections import Counter
from pyecharts import Bar,Line,Overlap,Grid,Page
import xgboost
import lightgbm
from sklearn import ensemble


class FeatureCountsPlot():
    '''特征与坏账统计。绘图。pyecharts'''
    def __init__(self):
        pass
    def onevar(self,data,x_index,y_index):
        '''统计函数
        param
            data：frame 数据源
            x_index：int 特征所在列索引
            y_index: int 目标所在列索引
        return
            A: frame
        '''            
        A=pd.DataFrame(data.groupby(data.columns[x_index],as_index=False).size())
        A.columns=['单量']
        B=data[data.iloc[:,y_index]=='1'].groupby(data.columns[x_index]).size()
        A['坏账']=B
        A=A.reset_index()
        A['好客户']=list(map(lambda x,y:x-y,A['单量'],A['坏账']))    
        A['坏账率']=list(map(lambda x,y:round(x/y,4),A['坏账'],A['单量']))
        A['总体坏账率']=round(A['坏账'].sum()/A['单量'].sum(),4)
        A=A.applymap(lambda x:x if pd.notnull(x) else 0)
        return A

    def drawfunc(self,A,text,yaxis_max=200000,yaxis_force_interval=0.25,width=1000):
        '''坏账率绘图函数'''
        bar = Bar(text,title_pos='center',width='100%')  
        line = Line(text,title_pos='center',width='100%') 
        bar.add("好客户",A.iloc[:,0].tolist(),A['好客户'].tolist(),yaxis_min=0,yaxis_max=yaxis_max,is_label_show=True,label_pos='inside',
                label_color=['#FFB6B9'],legend_pos='right', legend_orient='vertical',is_stack=True)
        bar.add("坏客户",A.iloc[:,0].tolist(),A['坏账'].tolist(),yaxis_min=0,yaxis_max=yaxis_max,is_label_show=True,label_pos='inside',
                label_color=['#BBDED6'],legend_pos='right',legend_orient='vertical',is_stack=True) 
        line.add('坏账率',A.iloc[:,0].tolist(),A['坏账率'].tolist(),yaxis_min=0,yaxis_max=1,yaxis_force_interval=yaxis_force_interval,is_smooth=True,
                legend_pos='right',point_symbol='circle',legend_orient='vertical',line_width =2,is_label_show =True)
        line.add('总体坏账率',A.iloc[:,0].tolist(),A['总体坏账率'].tolist(),yaxis_min=0,yaxis_max=1,yaxis_force_interval=yaxis_force_interval,is_smooth=True,
                legend_pos='right',legend_orient='vertical',line_width =2,line_type='dotted')
        overlap = Overlap(width='100%')
        overlap.add(bar)
        overlap.add(line,is_add_yaxis=True,yaxis_index=1) 
        grid=Grid(width=width)
        grid.add(overlap,grid_right='10%')
        return grid


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


def divide_x_dtype(df):
    """
    划分变量数据类型
    注：缺失值需为np.nan
    """
    int_X=[]
    float_X=[]
    category_X=[]
    for i,ds in df.iteritems():
        ds = ds[ds.notnull()]
        try:
            ds = ds.apply(float)
        except:
            category_X.append(i)
        finally:
            if i not in category_X:
                ds_re = ds.apply(lambda x: x%1)
                if ds_re.max()==0:
                    int_X.append(i)
                else:
                    float_X.append(i)
    return {'int':int_X,'float':float_X,'category':category_X}

def null_value_counts(df,value=0.9):
    """
    变量缺失值统计
    缺失比例一般达到90%则可舍弃，也可以将值替换为有值(1)和空值(0)，继续作为特征
    param:
          df
          value:阈值，返回缺失比例>=该值的特征名,默认0.1
    return:
          缺失统计表
          比例较高特征列表
    """
    df_null=df.isnull().sum().sort_values(ascending=False)
    df_null_percent=df_null/df.shape[0]
    null_df=pd.concat([df_null,df_null_percent],axis=1)
    null_df.rename(columns={0:'null_nums',1:'null_percent'},inplace=True)
    null_df_style=null_df[null_df.null_nums>0]
    return null_df_style.style.bar(), list(null_df[null_df.null_percent>=value].index)

def vif_compute(data,index):
    '''方差膨胀因子(VIF)：大于10存在严重多重共线性
    data:frame 自变量
    index:被解释变量的索引位置'''
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    return variance_inflation_factor(data.values, index)

def feature_importance_ensemble(x_array,y_array,xName_list,n_estimators=100, max_depth=3,figsize=(22,8)):
    '''集成模型特征重要性计算、排序和绘图'''
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