import pandas as pd
import numpy as np
import datetime
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
matplotlib.rcParams['font.family']='STSong' # 图表中文显示
matplotlib.rcParams['font.size']=20

def date_transform(date):
    trans_date=datetime.datetime.strftime(datetime.datetime.strptime(str(date),'%Y%m')+datetime.timedelta(days=75),'%Y%m%d')
    return int(trans_date)

def drop_unfull_data(data):
    """删除表现期不完全月份数据"""
    drop_ser = pd.Series(list(map(lambda x,y:True if y<date_transform(x) else False,data['LEND_MONTH'],data['BUSI_DT'])))
    data = data.reset_index(drop=True)
    data = data.drop(drop_ser[drop_ser].index)
    return data.reset_index(drop=True)

def busiDt_to_mob(ratio):
    """观察标准转换(业务日期->还款周期，空值缩进)"""
    ratio_T = pd.DataFrame()
    for name,val in ratio.iterrows():
        drop_ser=pd.Series(list(map(lambda x:True if x<date_transform(name) else False,val.index)),index=val.index)
        val = val.drop(drop_ser[drop_ser].index)
        val.index = ['mob'+str(index+2) for index in range(len(val.index))]
        ratio_T  = pd.concat([ratio_T,val],axis = 1)
    ratio_T = ratio_T.T[['mob'+str(index+2) for index in range(len(ratio.columns))]]
    return ratio_T

def vintage_compute(data,div_column):
    """M2+率账龄计算(按月)"""
    loan_sum = data.query(("BUSI_DT=={} and DIV_COLUMN=='{}'".format(data['BUSI_DT'].max(),div_column)))[['LEND_MONTH','DIV_COLUMN','SUM_LOAN_AMT']]
    loan_sum = loan_sum.sort_values('LEND_MONTH')
    M2_plus_sum = data.query(("DIV_COLUMN=='{}'".format(div_column)))[['LEND_MONTH','DIV_COLUMN','BUSI_DT','M2P_AMT']]
    M2_plus_sum2= drop_unfull_data(M2_plus_sum)
    M2_plus_pivot = M2_plus_sum2.pivot_table(values='M2P_AMT',index='LEND_MONTH',columns='BUSI_DT',aggfunc=sum)
    merge_data =loan_sum.merge(M2_plus_pivot.reset_index(),left_on='LEND_MONTH',right_on='LEND_MONTH',how='left')
    ratio = pd.concat((merge_data['LEND_MONTH'],merge_data[M2_plus_pivot.columns.tolist()].div(merge_data['SUM_LOAN_AMT'],axis=0)),axis=1)
    ratio = ratio.set_index('LEND_MONTH')
    return merge_data[['LEND_MONTH','DIV_COLUMN','SUM_LOAN_AMT']].merge(ratio.reset_index(),on='LEND_MONTH'), busiDt_to_mob(ratio), merge_data

def vintage_plot(ratio_T,title,ylim=(0,0.2),save=False):
    """M2+率账龄绘图(按月)"""
    plt.figure(figsize=(15,6))
    for i in ratio_T.index:
        plt.plot(ratio_T.loc[i].index,ratio_T.loc[i].values,'o-')
    plt.legend(ratio_T.index,ncol=6,loc=9)
    plt.ylim(ylim[0],ylim[1])
    plt.title(title,fontsize=18)
    if save:
        plt.savefig(fname=title+'.png',dpi=300,bbox_inches='tight')

def total_vintage(amt_merge,title):
    """M2+率账龄计算(整体，不区分月)"""
    amt_merge = amt_merge.set_index('LEND_MONTH')
    amt_merge_T = busiDt_to_mob(amt_merge.iloc[:,2:])
    amt_merge_T2=pd.concat([amt_merge.iloc[:,:2],amt_merge_T],axis=1)
    m2_plus_total = pd.DataFrame(columns=['SUM_LOAN_AMT'+'_'+title,'m2_plus_ratio'+'_'+title])
    for i in amt_merge_T2.columns[2:]:
        st = amt_merge_T2[['SUM_LOAN_AMT',i]].dropna().sum(axis=0)
        m2_plus_total.loc[i,'m2_plus_ratio'+'_'+title] = st[i]/st['SUM_LOAN_AMT']
        m2_plus_total.loc[i,'SUM_LOAN_AMT'+'_'+title] = st['SUM_LOAN_AMT']
    return m2_plus_total

def total_vintage_plot(t_t,title,legend=[],maxOffset=0,minOffset=0,ylim=(0,0.12),save=False):
    """M2+率账龄绘图(整体，不区分月)"""
    plt.figure(figsize=(12,6))
    plt.plot(t_t.index,t_t.iloc[:,1].values,'o-',t_t.index,t_t.iloc[:,3].values,'o-',t_t.index,t_t.iloc[:,5].values,'o-')
    plt.legend(legend)
    plt.title(title,fontsize=20)
    plt.ylim(ylim[0],ylim[1])
    for i in t_t.index:
    #     for c in [1,3,5]:
    #         y = t_t.loc[i].iloc[c]-(0.01 if c==1 else -0.01 if c==3 else 0) # 偏移数值标签1:整体上移或下移
        ylst = []
        for c in [1,3,5]:
           ylst.append(t_t.loc[i].iloc[c])
        # 偏移数值标签2：同x轴上，最大值上移，最小值下移
        ylst[ylst.index(max(ylst))] = max(ylst)+maxOffset
        ylst[ylst.index(min(ylst))] = min(ylst)-minOffset
        for k,c in enumerate([1,3,5]):
            plt.text(i,ylst[k],'%.1f%%'%(t_t.loc[i].iloc[c]*100),horizontalalignment='center',fontsize=16)
    if save:
        plt.savefig(title+'.png',dpi=300,bbox_inches='tight')


if __name__=='__main__':
    sql_vintageM2Plus = """
        select 
        a.lend_month,
        b.busi_dt，
        b.M2P_AMT,
        a.DIV_COLUMN,
        a.sum_loan_amt
        from (
            select substr(a.apply_dt,1,6) lend_month,
                    case when a.STR_NO not in (select distinct str_no from gd_str_loanno) then '非挂单门店单'
                        when a.STR_NO in (select distinct str_no from gd_str_loanno) and a.LOAN_NO not in (select loan_no from gd_str_loanno) then '挂单门店主营单'
                        when a.LOAN_NO in (select loan_no from gd_str_loanno) then '挂单门店非主营单'
                        else '其他' end as DIV_COLUMN,
                    sum(a.loan_amt) sum_loan_amt
            from pclods.i_loan_base_info_v_new a
            where a.prod_line='PL201' and a.aprov_result='013005'
                    and a.apply_dt>='20180701' and a.apply_dt<'20190701'
            group by substr(a.apply_dt,1,6),
                    case when a.STR_NO not in (select distinct str_no from gd_str_loanno) then '非挂单门店单'
                            when a.STR_NO in (select distinct str_no from gd_str_loanno) and a.LOAN_NO not in (select loan_no from gd_str_loanno) then '挂单门店主营单'
                            when a.LOAN_NO in (select loan_no from gd_str_loanno) then '挂单门店非主营单'
                            else '其他' end
            ) a
        left join
            (select substr(a.apply_dt,1,6) lend_month,
                    b.busi_dt busi_dt,
                    case when a.STR_NO not in (select distinct str_no from gd_str_loanno) then '非挂单门店单'
                        when a.STR_NO in (select distinct str_no from gd_str_loanno) and a.LOAN_NO not in (select loan_no from gd_str_loanno) then '挂单门店主营单'
                        when a.LOAN_NO in (select loan_no from gd_str_loanno) then '挂单门店非主营单'
                        else '其他' end as DIV_COLUMN,
                    sum(b.remain_prin_amt) M2P_AMT
            from pclods.i_loan_base_info_v_new a 
            right join 
                (
                    select busi_dt,loan_no,remain_prin_amt
                    from pcl.bus_t_loan_overdue_h_v_new
                    where busi_dt in ('20181001','20181101','20181201','20190101','20190201','20190302', '20190401','20190501','20190601','20190701','20190801','20190901')
                        and overdue_level>=2 and deal_status='046001'
                ) b on b.loan_no=a.loan_no
            where a.prod_line='PL201' and a.aprov_result='013005'
                    and a.apply_dt>='20180701' and a.apply_dt<'20190701'
            group by substr(a.apply_dt,1,6),b.busi_dt,
                        case when a.STR_NO not in (select distinct str_no from gd_str_loanno) then '非挂单门店单'
                            when a.STR_NO in (select distinct str_no from gd_str_loanno) and a.LOAN_NO not in (select loan_no from gd_str_loanno) then '挂单门店主营单'
                            when a.LOAN_NO in (select loan_no from gd_str_loanno) then '挂单门店非主营单'
                            else '其他' end 
            ) b 
        on a.lend_month = b.lend_month and a.DIV_COLUMN = b.DIV_COLUMN"""

    data = pd.read_excel('./账龄数据.xlsx')  # 账龄数据.xlsx为sql_vintageM2Plus提取的数据
    a,b,c1=vintage_compute(data=data,div_column='挂单门店主营单') 
    vintage_plot(ratio_T=b,title='挂单门店主营单',ylim=(0,0.15),save=False)
    a,b,c2=vintage_compute(data=data,div_column='挂单门店非主营单') 
    vintage_plot(ratio_T=b,title='挂单门店非主营单',ylim=(0,0.15),save=False)
    a,b,c3=vintage_compute(data=data,div_column='非挂单门店单') 
    vintage_plot(ratio_T=b,title='非挂单门店单',ylim=(0,0.15),save=False)
    t1 = total_vintage(c1,title='挂单门店主营单')
    t2 = total_vintage(c2,title='挂单门店非主营单')
    t3 = total_vintage(c3,title='非挂单门店单')
    t_t = pd.concat([t1,t2,t3],axis=1)
    total_vintage_plot(t_t,title='各还款周期M2+率',legend=['挂单门店主营单','挂单门店非主营单','非挂单门店单'],maxOffset=0.007,minOffset=0.005,ylim=(0,0.12),save=False)