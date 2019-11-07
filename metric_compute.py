import pandas as pd 
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score,f1_score,recall_score


class MetricKsRoc():
    '''计算ks值和auc值，并绘制ks和roc曲线'''
    def __init__(self,y_true,y_score):
        self.y_true = y_true
        self.y_score = y_score
        self.tpr = None
        self.fpr = None
        self.ks_value = None
        self.auc_value = None
        
    def compute_tpr_fpr_ks(self):
        """
        Compute TPR, FPR, precision, specificity, f1_score, accuarcy, thresholds.
        print auc value, ks value
        """
        tpr = []
        fpr = []
        for i in np.arange(0, 1.00001, 0.001):
            y_pred = self.y_score > i
            tp = (self.y_true * y_pred).sum()
            fn = (self.y_true * (1 - y_pred)).sum()
            fp = ((1 - self.y_true) * y_pred).sum()
            tn = ((1 - self.y_true) & (1 - y_pred)).sum()
            tpr_ = tp / (tp + fn)
            fpr_ = fp / (fp + tn)
            tpr.append(tpr_)
            fpr.append(fpr_)
        self.tpr = np.array(tpr)
        self.fpr = np.array(fpr)
        self.ks_value = abs(self.fpr - self.tpr).max()
        self.auc_value = roc_auc_score(self.y_true,self.y_score)
        print(f'AUC value is {self.auc_value}.')
        print(f'KS value is {self.ks_value}.')
    
    def plot_roc_curve(self,ax=None,title='ROC Curve'):
        """
        Plot ROC curve.
        """
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(7, 6))
        lw = 1.5
        ax.plot(self.fpr, self.tpr, color='darkblue',
                 lw=lw, label='ROC curve (area = %0.4f)' % self.auc_value)
        ax.plot([0, 1], [0, 1], color='darkred', lw=lw, linestyle='--')
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        return ax
    
    def plot_ks_curve(self,ax=None,title='KS Curve'):
        """
        Plot KS curve.
        """
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(7, 6))
        # compute the position of ks value
        ks = 0
        best_th = 0
        upper_end = 0
        lower_end = 0
        thresholds = [i for i in np.arange(0, 1.00001, 0.001)]
        for tpr_, fpr_, th_ in zip(self.tpr, self.fpr, thresholds):
            ks_ = abs(tpr_ - fpr_)
            if ks_ > ks:
                ks = ks_
                best_th = th_
                upper_end = max(tpr_, fpr_)
                lower_end = min(tpr_, fpr_)
        # plot ks curve
        lw = 1.5
        ax.plot(sorted(thresholds, reverse=True), self.tpr, color='darkblue',
                 lw=lw, label='True Positive Rate')
        ax.plot(sorted(thresholds, reverse=True), self.fpr, color='darkgreen',
                 lw=lw, label='False Positive Rate')
        ax.plot([1 - best_th, 1 - best_th], [lower_end + 0.01, upper_end - 0.01],
                color='darkred', lw=lw, linestyle='--',
                label='KS value: {:.4f}'.format(self.ks_value))
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel('Probability Threshold')
        ax.set_ylabel('True/False Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        return ax