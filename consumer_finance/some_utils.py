import pandas as pd
import numpy as np
import os
import pymysql
import cx_Oracle
import math

# 有利反欺诈库
mysql_configs = {
                'ip':'119.254.115.72',
                'port':3307,
                'user':'haoran.yang',
                'password':'1qaz@wsx',
                'dbname':'fri'
                }
# 快信消金库
oracle_configs = {
                'username': 'haoran_yang_pohoocredit',
                'password': 'yhr12#',
                'host': '10.30.4.26',
                'port': '1521',
                'service_name': 'odsdb'
                }

# 需要转换码值的字段
trans_cols = ['cust_open_org', 'aprov_result', 'aprov_decision', 'is_insuuance', 'subj_id', 'is_attach', 'is_sa_rufuse', 'is_back_to_sa', 'is_back_to_check', 
              'chal_code', 'fund_channel','cert_type', 'chal_code', 'cust_type', 'sex', 'reg_type', 'is_reg_live', 'live_build_type', 
              'mth_tel_bill', 'is_real_name', 'mar_status', 'unit_type', 'industry', 'unit_scale', 'education', 'is_loaned', 'settle_type']

def get_data_from_mysql(sql,configs=mysql_configs):
    connection=pymysql.connect(host=configs['ip'], port=configs['port'], user=configs['user'],
                                password=configs['password'], db=configs['dbname'], charset='utf8mb4')
    with connection.cursor() as cursor:
        cursor.execute(sql)
        df=cursor.fetchall()
    return df

def get_data_from_oracle(sql,configs=oracle_configs):
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

def insert_data_to_oracle(sql,dataLst,configs=oracle_configs):
    '''数据批量插入。
    sql示例: INSERT INTO table_name(loan_no, fraudScore) VALUES(:loan_no,:fraudScore)
    dataLst示例: [{'loan_no': '15000201801010475', 'fraudScore': '300'},
                  {'loan_no': '15000201801010407', 'fraudScore': '40'}]
    dataframe转换dataLst方式：[v for k,v in datas.T.to_dict().items()]
    '''
    oracle_tns = cx_Oracle.makedsn(oracle_configs.get('host'), oracle_configs.get('port'), oracle_configs.get('service_name'))
    connectObj = cx_Oracle.connect(oracle_configs.get('username'), oracle_configs.get('password'), oracle_tns)
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

if __name__=='__main__':
    base_info = get_data_from_oracle(sql='''select * from PCLODS.I_LOAN_BASE_INFO_V_NEW t where rownum<=100''')
    sys_code = get_data_from_oracle(sql='''select * from PCL.SYS_CODE_INFO t''')
    data,replace_dict = code_transform(data=base_info,sys_code=sys_code)