import pandas as pd
import numpy as np
import copy


#方案一
class EntropyGiniBinning():
    def __init__(self):
        pass

    def entropy_dt(self, data, target):
        '''信息熵'''
        entropy_init = 0
        for k in list(data[target].unique()):
            Pk = data[data[target]==k].shape[0]/data.shape[0]
            entropy_init = entropy_init + Pk*np.log2(Pk)
        return -entropy_init

    def info_gain(self, data, feature, target):
        '''信息增益'''
        total_entropy = self.entropy_dt(data, target)
        entLoss_init = 0
        for v in list(data[feature].unique()):
            data_t = data[data[feature]==v]
            ent_t = self.entropy_dt(data = data_t, target=target)
            entLoss_init = entLoss_init + data_t.shape[0]/data.shape[0]*ent_t
        return total_entropy - entLoss_init

    def gain_ratio(self, data, feature, target):
        '''增益率'''
        inf_g = self.info_gain(data,feature,target)
        iv = 0
        for v in list(data[feature].unique()):
            data_t = data[data[feature]==v]
            iv_t = (data_t.shape[0]/data.shape[0])*np.log2(data_t.shape[0]/data.shape[0])
            iv = iv + iv_t
        return inf_g/(-iv)

    def gini_dt(self, data, feature, target):
        '''基尼指数'''
        def gini_count(data, target):
            gini_init = 0
            for k in list(data[target].unique()):
                Pk = data[data[target]==k].shape[0]/data.shape[0]
                gini_init = gini_init + Pk**2
            return 1-gini_init
        gini_feature = 0
        for v in list(data[feature].unique()):
            data_t = data[data[feature]==v]
            gini_f_t = gini_count(data=data_t, target=target)
            gini_feature = gini_feature + data_t.shape[0]/data.shape[0]*gini_f_t
        return gini_feature

    def bi_partition(self, data,feature,target,min_samples_leaf):
        '''基于信息增益/基尼指数采用二分法寻找单个最优划分点'''
        data_t = data.copy(deep=True)
        values = list(data[feature].unique())
        values.sort() #取值排序
        split_vals = []
        gini_dict = dict()
        for i in range(1,len(values)):
            split_vals.append((values[i-1]+values[i])/2)  # 生成划分点
        for split_val in split_vals: #遍历划分点
            data_t['bi_cut'] = data_t[feature].apply(lambda x: 0 if x<= split_val else 1)
            left_len = data_t[data_t['bi_cut']==0].shape[0] 
            right_len = data_t[data_t['bi_cut']==1].shape[0]
            if left_len<min_samples_leaf or right_len<min_samples_leaf: #划分后样本数小于指定值，则停止划分
                pass
            else:
                # gini_dict[split_val] = self.info_gain(data_t,feature='bi_cut',target=target) #基于信息增益
                gini_dict[split_val] = self.gini_dt(data_t,feature='bi_cut',target=target) #基于基尼指数
        if gini_dict:
            # return max(gini_dict,key=gini_dict.get)
            return min(gini_dict,key=gini_dict.get) #返回基尼指数最小划分点
        else:
            return None

    def best_split_def(self, data, feature, target, min_samples_leaf):
        '''寻找最优划分点组合。缺点：响应速度太慢'''
        best_splist_list = []
        data_cut = [data]
        data_len = 1
        while True:
            i = 0
            data_c = data.copy(deep=True)
            # 遍历每个数据集，只要找一个划分点则停止遍历，根据现有划分点切割成新的数据集，重新遍历，直到不能在任何一个数据集上找到新的划分点
            for data_t in data_cut:
                # 寻找单个最优划分点
                best_split = self.bi_partition(data=data_t,feature=feature,target=target,min_samples_leaf=min_samples_leaf)
                if best_split is not None:
                    best_splist_list.append(best_split)
                    best_splist_list2 = copy.deepcopy(best_splist_list)
                    best_splist_list2.sort() # 对划分点排序
                    # 切割生成新的数据集
                    data_cut = []
                    for split_p in best_splist_list2:
                        data_cut.append(data_c[data_c[feature]<=split_p])
                        data_c = data_c.drop(data_c[data_c[feature]<=split_p].index)
                    data_cut.append(data_c)
                    data_len = len(data_cut)
                    break # 跳出for循环重新遍历新的数据集
                i+=1
            if i==data_len: #遍历完整个数据集则跳出while循环
                break
        return best_splist_list


#方案二
'''get_bestsplit_list方法与EntropyGiniBinning类中split_numeric方法结果一致，但响应更快。'''
def calc_score_median(sample_set, var):
    '''
    计算相邻评分的中位数，以便进行决策树二元切分
    param sample_set: 待切分样本
    param var: 分割变量名称
    '''
    var_list = list(np.unique(sample_set[var]))
    var_median_list = []
    for i in range(len(var_list) -1):
        var_median = (var_list[i] + var_list[i+1]) / 2
        var_median_list.append(var_median)
    return var_median_list

def choose_best_split(sample_set, var, target,min_sample):
    '''
    使用CART分类决策树选择最好的样本切分点
    返回切分点
    param sample_set: 待切分样本
    param var: 分割变量名称
    param min_sample: 待切分样本的最小样本量(限制条件)
    '''
    # 根据样本评分计算相邻不同分数的中间值
    score_median_list = calc_score_median(sample_set, var)
    median_len = len(score_median_list)
    sample_cnt = sample_set.shape[0]
    sample1_cnt = sum(sample_set[target])
    sample0_cnt =  sample_cnt- sample1_cnt
    Gini = 1 - np.square(sample1_cnt / sample_cnt) - np.square(sample0_cnt / sample_cnt)

    bestGini = 0.0; bestSplit_point = 0.0; bestSplit_position = 0.0
    for i in range(median_len):
        left = sample_set[sample_set[var] < score_median_list[i]]
        right = sample_set[sample_set[var] > score_median_list[i]]

        left_cnt = left.shape[0]; right_cnt = right.shape[0]
        left1_cnt = sum(left[target]); right1_cnt = sum(right[target])
        left0_cnt =  left_cnt - left1_cnt; right0_cnt =  right_cnt - right1_cnt
        left_ratio = left_cnt / sample_cnt; right_ratio = right_cnt / sample_cnt

        if left_cnt < min_sample or right_cnt < min_sample:
            continue
        Gini_left = 1 - np.square(left1_cnt / left_cnt) - np.square(left0_cnt / left_cnt)
        Gini_right = 1 - np.square(right1_cnt / right_cnt) - np.square(right0_cnt / right_cnt)
        Gini_temp = Gini - (left_ratio * Gini_left + right_ratio * Gini_right)
        if Gini_temp > bestGini:
            bestGini = Gini_temp; bestSplit_point = score_median_list[i]
            if median_len > 1:
                bestSplit_position = i / (median_len - 1)
            else:
                bestSplit_position = i / median_len
        else:
            continue
    Gini = Gini - bestGini
    return bestSplit_point, bestSplit_position

def bining_data_split(sample_set, var, target, min_sample, split_list):
    '''
    划分数据找到最优分割点list
    param sample_set: 待切分样本
    param var: 分割变量名称
    param min_sample: 待切分样本的最小样本量(限制条件)
    param split_list: 最优分割点list
    '''
    split, position = choose_best_split(sample_set, var,target, min_sample)
    if split != 0.0:
        split_list.append(split)
    # 根据分割点划分数据集，继续进行划分
    sample_set_left = sample_set[sample_set[var] < split]
    sample_set_right = sample_set[sample_set[var] > split]
    # 如果左子树样本量超过2倍最小样本量，且分割点不是第一个分割点，则切分左子树
    if len(sample_set_left) >= min_sample * 2 and position not in [0.0, 1.0]:
        bining_data_split(sample_set_left, var, target, min_sample, split_list)
    else:
        None
    # 如果右子树样本量超过2倍最小样本量，且分割点不是最后一个分割点，则切分右子树
    if len(sample_set_right) >= min_sample * 2 and position not in [0.0, 1.0]:
        bining_data_split(sample_set_right, var, target, min_sample, split_list)
    else:
        None

def get_bestsplit_list(sample_set, var, target,threshold=0.05,sort=False,for_cut=False):
    '''
    根据分箱得到最优分割点list
    param sample_set: 待切分样本
    param var: 分割变量名称
    target:目标变量名
    threshold:最小样本阈值，默认0.05，可取整数
    '''
    # 计算最小样本阈值（终止条件）
    if threshold<1:
        min_df = int(sample_set.shape[0] * threshold)
    else:
        min_df = threshold
    print('min sample split is %s'%min_df)
    split_list = []
    # 计算第一个和最后一个分割点
    bining_data_split(sample_set, var, target, min_df, split_list)
    if split_list:
        if sort:
            split_list.sort()
        if for_cut and sort:
            split_list.insert(0,float('-inf'))
            split_list.append(float('inf'))
    return split_list


# 方案三。R smbinning最优分箱。
"""
# 加载所需库
library(grid)
library(partykit)
library(libcoin)
# library(mvnorm)
library(rpart)
library(Formula)
library(smbinning)
setwd("F:\\欺诈模型")

# 读入数据
dat = read.csv('numb_datas_for_Rbinning.txt',sep='~',header = T)  

# 定义最小划分阈值
p_val = 0.008
# 指定目标名
target = 'y_m3Worse'

# 目标列"y"转换为整数型,y取值需为0,1
dat[,target] = as.numeric(as.character(dat[,target])) 

d = list()  # 定义存储分割点列表
nms = names(dat)
for (i in nms[-which(nms==target)]) {# 遍历需要分箱的列名, names(dat)[1:4], c('car_loan','age','income')
    tem1 = dat[dat[, i] != -99999, ]  # 缺失值为-99999，包含0
    tem2 = dat[(dat[, i] != -99999) & (dat[, i] != 0), ]  # 缺失值为-99999，不含0
    print(i)  # 打印列名
    if ("出现错误" %in% tryCatch(smbinning(df = tem1, y = target, x = i, p = p_val), error = function(e) {
        print("出现错误")
    })) 
        next # 跳出循环
    if ("try-error" %in% class(try(smbinning(df = tem1, y = target, x = i, p = p_val)$ivtable, 
        silent = TRUE))) {
        print("try-error")
        result = smbinning(df = tem2, y = target, x = i, p = p_val)
        print('tem2')
        x.inv = try(result$ivtable, silent = TRUE)
        if ("try-error" %in% class(x.inv)) 
            next # 跳出循环
        # print(result$cuts)
        d[[i]] <- paste(c(0, result$cuts), collapse = ",")
        
    } else { 
        result = smbinning(df = tem1, y = target, x = i, p = p_val)
        print('tem1')
        x.inv = try(result$ivtable, silent = TRUE)
        if ("try-error" %in% class(x.inv)) 
            next # 跳出循环
        # print(result$ivtable) print(result$cuts)
        d[[i]] <- paste(result$cuts, collapse = ",")
    }
}
print(d)
write.table(data.frame(d),'R_smbinning.txt',sep = '~')  # 保存分箱点到本地

# smbinning.plot(result,options = 'WoE',sub = 'CreditAmount')
for (i in names(dat)[0:1]) {
    print(i)
    # if('出现错误' %in% tryCatch(smbinning(df = tem1,y='y',x=i,p=0.1),warning =
    # function(w){print('出现警告')},error=function(e){print('出现错误')} )) next
    if ("出现错误" %in% tryCatch(smbinning(df = tem1, y = target, x = i, p = 0.1), error = function(e) {
        print("出现错误")
    })) 
        next
    result = smbinning(df = tem1, y = target, x = i, p = 0.1)
    x.inv = try(result$ivtable, silent = TRUE)
    if ("try-error" %in% class(x.inv)) 
        next
    # print(result$ivtable)
    print(result$cuts)
}
"""