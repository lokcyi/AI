import sweetviz
import pandas as pd
import numpy as np
from datetime import datetime

def get_diff_time(cols):
    s_date = cols[0]
    e_date = cols[1]
    diff_time = e_date - s_date
    # cycletime = math.ceil((diff_time.total_seconds()/3600)) 
    # cycletime = diff_time.astype('timedelta64[D]')
    days = diff_time.days
    seconds = diff_time.seconds
    hours = seconds/3600
    if hours >=12 :
      cycletime = days + 0.5
    else:
      cycletime = days
    return cycletime

train_select_columns = ['DATA_DATE','LOT_ID','STATUS','CHIPNAME','LAYER','REMAIN_LAYER_SEQ',
  'OP_NO','REMAIN_OP_SEQ','PRIORITY','LOT_TYPE','WIP_QTY','WS_DATE','IS_MAIN_ROUTE','ACTUAL_WP_OUT']

#train_select_columns = ['DATA_DATE','STATUS','CHIPNAME','LAYER','IS_MAIN_ROUTE','ACTUAL_WP_OUT']


train_raw_data = pd.read_csv('D:/projects/ai/poc/homework/training_data_20210128.csv',usecols=train_select_columns)
test_raw_data = pd.read_csv('D:/projects/ai/poc/homework/testing_data_20210128.csv',usecols=train_select_columns)


train_raw_data = pd.read_csv('data/Training_data_20210128.csv' ,usecols=train_select_columns)
test_raw_data = pd.read_csv('data/testing_data_20210128.csv',usecols=train_select_columns)
# train_raw_data= train_raw_data.dropna()
# test_raw_data= test_raw_data.dropna()

print(test_raw_data.info())
train_raw_data['DATA_DATE'] = pd.to_datetime(train_raw_data['DATA_DATE']) #, format='%Y%m%d')
# df['WS_DATE']=pd.to_datetime(df['WS_DATE'])
train_raw_data['ACTUAL_WP_OUT']=pd.to_datetime(train_raw_data['ACTUAL_WP_OUT'])
test_raw_data['DATA_DATE'] = pd.to_datetime(test_raw_data['DATA_DATE']) #, format='%Y%m%d')
# df['WS_DATE']=pd.to_datetime(df['WS_DATE'])
test_raw_data['ACTUAL_WP_OUT']=pd.to_datetime(test_raw_data['ACTUAL_WP_OUT'])

train_raw_data['REMAIN_CYCLE_TIME'] = train_raw_data[['DATA_DATE','ACTUAL_WP_OUT']].apply(get_diff_time, axis=1) 
test_raw_data['REMAIN_CYCLE_TIME'] = test_raw_data[['DATA_DATE','ACTUAL_WP_OUT']].apply(get_diff_time, axis=1) 

feature_config=sweetviz.FeatureConfig(force_num=['REMAIN_CYCLE_TIME'])
my_report = sweetviz.compare([train_raw_data, "Train"], [test_raw_data, "Test"] , target_feat='REMAIN_CYCLE_TIME' ,feat_cfg= feature_config)
my_report.show_html("Report2.html")

