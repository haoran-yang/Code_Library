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