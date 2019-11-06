import pandas as pd
import numpy as np 
from best_binning import get_bestsplit_list
from data_preprocessing import FeatureLabelSizeRatio
from func_timeout import func_set_timeout, FunctionTimedOut # 超时设置函数

def bBinning_cut_woeReplace(data,cols,target,bins_dt,cut_bins=4,threshold=0.05):
    '''特征批量最优分箱并生成woe特征列。bins_dt:R最优分箱字典,键为特征名,值为分箱点列表。cut_bins:python分箱最大分箱点数。threshold:python gini分箱阈值'''
    tt_iv_dict = {} # 总IV
    bbinning_fets=[] # 成功分箱特征
    no_bbins_fets=[] # 分箱失败特征
    flsr=FeatureLabelSizeRatio()
    # 批量最优分箱
    for f in cols:
        bb_bins = bins_dt.get(f)
        if bb_bins is not None:
            # 1.R分箱
            print(f'R binning "{f}"')
            data[f+'_bbinning'] = pd.cut(data[f],bins=bb_bins)
            # 计算总IV
            tt_iv = flsr.get_woe_iv(df=data.astype({f+'_bbinning':str}), x=f+'_bbinning', x_type='cate', y=target, only_total_iv=True, style=False, sort=False)
            tt_iv_dict.update({f+'_bbinning':tt_iv})
            print(f'total iv is {tt_iv}')
            # 计算woe并替换
            woe_replace = flsr.get_woe_dict(df=data,x=f+'_bbinning',y=target,x_type='cate')
            data[f+'_bbinning_woe'] = data[f+'_bbinning'].replace(woe_replace)
            bbinning_fets.append(f+'_bbinning')
            print('-'*40)
        else:
            # 2.python gini分箱
            try:
                py_bins = get_bestsplit_list(sample_set=data[data[f].notnull()][[f,target]],target=target,var=f,threshold=threshold,sort=False,for_cut=False)
            except FunctionTimedOut as e:
                print('python gini分箱超时.')
                py_bins = []
            if py_bins:
                del_bins = len(py_bins) - cut_bins
                if del_bins>0:
                    py_bins = py_bins[0:-del_bins] # 删除多余的分箱点
                py_bins.sort(); py_bins.insert(0, float('-inf')); py_bins.append(float('inf'))
                print(f'Python gini binning "{f}"')
                data[f+'_bbinning'] = pd.cut(data[f],bins=py_bins)
                tt_iv = flsr.get_woe_iv(df=data.astype({f+'_bbinning':str}), x=f+'_bbinning', x_type='cate', y=target, only_total_iv=True, style=False, sort=False)
                tt_iv_dict.update({f+'_bbinning':tt_iv})
                print(f'total iv is {tt_iv}')
                woe_replace = flsr.get_woe_dict(df=data,x=f+'_bbinning',y=target,x_type='cate')
                data[f+'_bbinning_woe'] = data[f+'_bbinning'].replace(woe_replace)
                bbinning_fets.append(f+'_bbinning')
                print('-'*40)
            else:
                # 3.无分箱点
                no_bbins_fets.append(f)
                print(f'"{f}" no best bins!')
                print('-'*40)
    return tt_iv_dict, bbinning_fets, no_bbins_fets

def cut_woeReplace(data,cols,target,bins_dt):
    '''批量分箱并woe替换。bins_dt: 分箱字典,键为特征名,值为分箱点列表'''
    tt_iv_dict = {}
    bbinning_fets=[]
    flsr=FeatureLabelSizeRatio()
    # 分箱并woe替换
    for f in cols:
        bb_bins = bins_dt.get(f)
        if bb_bins is not None:
            # R分箱
            print(f'R binning "{f}"')
            data[f+'_bbinning'] = pd.cut(data[f],bins=bb_bins)
            tt_iv = flsr.get_woe_iv(df=data.astype({f+'_bbinning':str}), x=f+'_bbinning', x_type='cate', y=target, only_total_iv=True, style=False, sort=False)
            tt_iv_dict.update({f+'_bbinning':tt_iv})
            print(f'total iv is {tt_iv}')
            woe_replace = flsr.get_woe_dict(df=data,x=f+'_bbinning',y=target,x_type='cate')
            data[f+'_bbinning_woe'] = data[f+'_bbinning'].replace(woe_replace)
            bbinning_fets.append(f+'_bbinning')
            print('-'*40)
    return tt_iv_dict, bbinning_fets

def cate_orderLabel_byTarget(data,cols,target):
    '''取值数较多的类别特征, 按target正类比例从低到高的顺序进行标签赋值。
       编码后的新特征列与target高度相关,可结合分箱方法(如R最优分箱), 降低过拟合风险。'''
    orderLabel = pd.DataFrame(index=data.index)
    flsr=FeatureLabelSizeRatio()
    for f in cols:
        a,b = flsr.get_sampleSize_oneRatio(data,f,'cate',target,'loan_no',style=False)
        rep_d = {}
        i = 0
        for v in a.index:
            rep_d.update({v:i})
            i+=1
        orderLabel[f+'_ordLabel'] = data[f].replace(rep_d)
    return orderLabel