import pandas as pd
import numpy as np
import os
import pymysql
import cx_Oracle
import math

def get_data_from_mysql(sql,configs):
    connection=pymysql.connect(host=configs['ip'], port=configs['port'], user=configs['user'],
                                password=configs['password'], db=configs['dbname'], charset='utf8mb4')
    with connection.cursor() as cursor:
        cursor.execute(sql)
        df=cursor.fetchall()
    return df

def get_data_from_oracle(sql,configs):
    os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8' # 'SIMPLIFIED CHINESE_CHINA.UTF8'
    username = configs['username']
    password = configs['password']
    addr = '{}:{}/{}'.format(configs['host'], configs['port'], configs['service_name'])
    db = cx_Oracle.connect(username, password, addr)
    cr = db.cursor()
    cr.execute(sql)
    rx = cr.description
    df = cr.fetchall()
    cr.close()
    db.close()
    dfs=pd.DataFrame(df,columns=[i[0].lower() for i in rx])
    return dfs

def insert_data_to_oracle(sql,dataLst,configs):
    '''数据批量插入。
    sql示例: INSERT INTO table_name(Numbs, Scores) VALUES(:Numbs,:Scores)
    dataLst示例: [{'Numbs': '42004545401047', 'Scores': '300'},
                  {'Numbs': '67064645010407', 'Scores': '40'}]
    dataframe转换dataLst方式：[v for k,v in datas.T.to_dict().items()]
    '''
    oracle_tns = cx_Oracle.makedsn(configs.get('host'), configs.get('port'), configs.get('service_name'))
    connectObj = cx_Oracle.connect(configs.get('username'), configs.get('password'), oracle_tns)
    cursorObj = connectObj.cursor()
    cursorObj.prepare(sql)
    cursorObj.executemany(None, dataLst)
    connectObj.commit()
    cursorObj.close()

def code_transform(data,sys_code):
    """码值转换"""
    replace_dict = {}
    replace_data = pd.DataFrame()
    for col in trans_cols:
        if col in data.columns:
            replace_t = {}
            for val in data[col].unique():
                if pd.notnull(val):
                    code_df = sys_code[sys_code['code_info_value']==val]['code_info_name']
                    if code_df.shape[0]==1:
                        replace_t.update({val:code_df.iloc[0]})
                    elif code_df.shape[0]>1:
                        print('column:%s value:%s not only in sys_code_info.'%(col,val))
            if replace_t:
                replace_data[col] = data[col].replace(replace_t)
                replace_dict[col] = replace_t
    return replace_data,replace_dict


def data_cut(series_data, labels=False, cuts=20):
    data_of_cut=pd.cut(series_data,bins=cuts,labels=labels,include_lowest=True)
    datacounts=pd.value_counts(data_of_cut,sort=False)
    return datacounts

def get_psi(series_data1, series_data2):
    '''psi值计算'''
    data1=data_cut(series_data1)
    data2=data_cut(series_data2)
    data1_percent=data1.map(lambda x:x/data1.sum())
    data2_percent=data2.map(lambda x:x/data2.sum())
    total=pd.concat([data1_percent,data2_percent],axis=1)
    total['div']=total.iloc[:,0]/total.iloc[:,1]
    total['div_ln']=total['div'].apply(lambda x:math.log(x,math.e))
    total['signal_psi']=(total.iloc[:,0]-total.iloc[:,1])*total['div_ln']
    psi_data=total['signal_psi'].sum()
    return psi_data

def psi_oneByone(df, columnname):
    '''psi调用函数'''
    psi_dict={}
    for x in df.date.unique().tolist():
        psi_dict[x]={}
        for y in df.date.unique().tolist():
            data_one=df.query('date=="{}"'.format(x))[columnname]
            data_two=df.query('date=="{}"'.format(y))[columnname]
            psivalue=get_psi(data_one,data_two)
            psi_dict[x][y]=psivalue
    return pd.DataFrame(psi_dict)

if __name__=='__main__':
    # 读取需要转换码值的字段
    with open('./code_trans_cols.txt','r') as f:
        trans_cols = eval(f.read())
    base_info = get_data_from_oracle(sql='''select * from tablename t where rownum<=100''',configs={})
    sys_code = get_data_from_oracle(sql='''select * from tablename t''',configs={})
    data,replace_dict = code_transform(data=base_info,sys_code=sys_code)