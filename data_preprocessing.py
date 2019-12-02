import pandas as pd
import numpy as np
import time, datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import math
from collections import Counter
from pyecharts import Bar,Line,Overlap,Grid,Page #0.5.0

def build_cross_feats(data,cols):
    '''两两特征交叉,cols为特征名列表'''
    cross_dfs = pd.DataFrame(index=data.index)
    for i in range(len(cols)-1):
        for j in range(i+1,len(cols)):
            cross_dfs[cols[i]+'_'+cols[j]] = data[cols[i]]+'_'+data[cols[j]]
    return cross_dfs.fillna('nan_nan')

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

    def get_woe_dict(self,df,x,y,x_type):
        df_concat=df[[x,y]]    
        df_concat[y] = df_concat[y].apply(float)
        x_values=list(df_concat[x].unique())
        if x_type == 'numeric':
            x_values.sort()
        s_total=df_concat.groupby(y).size()
        woe_dic={}
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
            woe_dic[x_values[i]]=woe
        return woe_dic

def cycle_time_encode(data,cols):
    '''循环特征编码。对于月份、天，这种首尾相连的闭环特征，可以用正弦余弦来编码。'''
    data_t = data[cols].copy(deep=True)
    for col in cols:
        data_t[col+'_sin']=np.sin((2*np.pi*data_t[col])/max(data_t[col]))
        data_t[col+'_cos']=np.cos((2*np.pi*data_t[col])/max(data_t[col]))
    return data_t.drop(columns=cols)