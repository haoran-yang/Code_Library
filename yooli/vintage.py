import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt
import seaborn as sns

def date_transform(date):
    trans_date=datetime.datetime.strftime(datetime.datetime.strptime(str(date),'%Y%m')+datetime.timedelta(days=75),'%Y%m%d')
    return int(trans_date)

def drop_unfull_data(data):
    drop_ser = pd.Series(list(map(lambda x,y:True if y<date_transform(x) else False,data['LEND_MONTH'],data['BUSI_DT'])))
    data = data.reset_index(drop=True)
    data = data.drop(drop_ser[drop_ser].index)
    return data.reset_index(drop=True)

def vintage_compute(data,str_prov):
    loan_sum = data.query(("BUSI_DT=={} and STR_PROV=='{}'".format(data['BUSI_DT'].max(),str_prov)))[['LEND_MONTH','STR_PROV','SUM_LOAN_AMT']]
    loan_sum = loan_sum.sort_values('LEND_MONTH')
    M2_plus_sum = data.query(("STR_PROV=='{}'".format(str_prov)))[['LEND_MONTH','STR_PROV','BUSI_DT','M2P_AMT']]
    M2_plus_sum2= drop_unfull_data(M2_plus_sum)
    M2_plus_pivot = M2_plus_sum2.pivot_table(values='M2P_AMT',index='LEND_MONTH',columns='BUSI_DT',aggfunc=sum)
    merge_data =loan_sum.merge(M2_plus_pivot.reset_index(),left_on='LEND_MONTH',right_on='LEND_MONTH',how='left')
    ratio = pd.concat((merge_data['LEND_MONTH'],merge_data[M2_plus_pivot.columns.tolist()].div(merge_data['SUM_LOAN_AMT'],axis=0)),axis=1)
    ratio = ratio.set_index('LEND_MONTH')
    ratio_T = pd.DataFrame()
    for name,val in ratio.iterrows():
        drop_ser=pd.Series(list(map(lambda x:True if x<date_transform(name) else False,val.index)),index=val.index)
        val = val.drop(drop_ser[drop_ser].index)
        val.index = ['mob'+str(index+2) for index in range(len(val.index))]
        ratio_T  = pd.concat([ratio_T,val],axis = 1)
    ratio_T = ratio_T.T[['mob'+str(index+2) for index in range(len(ratio.columns))]]
    return merge_data[['LEND_MONTH','STR_PROV','SUM_LOAN_AMT']].merge(ratio.reset_index(),on='LEND_MONTH'), ratio_T

def vintage_plot(ratio_T,title,ylim=(0,0.2),save=False):
    sns.set()
    plt.figure(figsize=(15,6))
    for i in ratio_T.index:
        plt.plot(ratio_T.loc[i].index,ratio_T.loc[i].values,'o-')
    plt.legend(ratio_T.index,ncol=6,loc=9)
    plt.ylim(ylim[0],ylim[1])
    plt.title(title,fontsize=18)
    if save:
        plt.savefig(fname=title+'.png',dpi=300,bbox_inches='tight')

if __name__=='__main__':
    sql_vintageM2Plus = """select 
                            a.lend_month,
                            b.busi_dt，
                            b.M2P_AMT,
                            a.str_prov,  --区分新疆和其他省
                            a.sum_loan_amt
                                from (
                                    select substr(a.apply_dt,1,6) lend_month,
                                            case when a.STR_PROV='新疆维吾尔自治区' then '新疆' else '其他省' end as str_prov,  --区分新疆和其他省
                                            sum(a.loan_amt) sum_loan_amt
                                    from pclods.i_loan_base_info_v_new a
                                    where a.prod_line='PL201' and a.aprov_result='013005'
                                            and a.apply_dt>='20180701' and a.apply_dt<'20190101'
                                            and a.LOAN_NO not in (select loan_no from qz_bj_loan_no)   --限制条件：非欺诈单
                                    group by substr(a.apply_dt,1,6),
                                            case when a.STR_PROV='新疆维吾尔自治区' then '新疆' else '其他省' end  --区分新疆和其他省
                                    ) a
                                left join
                                    (select substr(a.apply_dt,1,6) lend_month,
                                            b.busi_dt busi_dt,
                                            case when a.STR_PROV='新疆维吾尔自治区' then '新疆' else '其他省' end as str_prov,  --区分新疆和其他省
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
                                            and a.LOAN_NO not in (select loan_no from qz_bj_loan_no)  --限制条件：非欺诈单
                                    group by substr(a.apply_dt,1,6),b.busi_dt,
                                            case when a.STR_PROV='新疆维吾尔自治区' then '新疆' else '其他省' end  --区分新疆和其他省
                                ) b 
                                on a.lend_month = b.lend_month and a.STR_PROV = b.STR_PROV"""

    data = pd.read_excel('./账龄数据.xlsx')  # 账龄数据.xlsx为sql_vintageM2Plus提取的数据
    a,b=vintage_compute(data=data,str_prov='新疆') 
    vintage_plot(ratio_T=b,title='XinJiang',ylim=(0,0.25))