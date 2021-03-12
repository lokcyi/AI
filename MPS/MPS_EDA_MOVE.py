# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np 
import math
import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.model_selection import GridSearchCV 
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn import metrics
import re
import sweetviz as sv
import datetime
final_date = '2021-01-16'



# %%
def trainLR(df):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    import joblib

    X,Y = preHandleDat(df,True)
    
    #拆分train validation set
    X_train, X_test,y_train,y_test = train_test_split(X,Y,test_size =0.2,random_state=587)
    # print the data
    model = LinearRegression(fit_intercept=True, normalize=True, copy_X=True) #fit_intercept=False
   
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    #print(y_predict)
    
    # 拟合优度R2的输出方法
    print("r2:", model.score(X_test, y_test))
    # 用Scikit_learn计算MAE
    print("MAE:", metrics.mean_absolute_error(y_test, y_predict))
    # 用Scikit_learn计算MSE
    print("MSE:", metrics.mean_squared_error(y_test, y_predict))
    
    # 用Scikit_learn计算RMSE
    print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_predict)))

    print("intercept_ :",model.intercept_)
    # print("coef_", model.coef_)
    joblib.dump(model, 'LR_model')

    return y_predict


# %%
def testLR(df):
    from sklearn.linear_model import LinearRegression
    import joblib
    X_test,y_test = preHandleDat(df,False)
    # print(X_test)
    #print(X_test,y_test)
    loaded_model = joblib.load('LR_model')
    y_predict = loaded_model.predict(X_test)
    df['predict'] = y_predict
    
    loaded_model.score(X_test, y_test)
    print("r2:", loaded_model.score(X_test, y_test))
    # y_test real 与 y_predict的可视化
    # 设置图片尺寸
    plt.figure(figsize=(10, 6))
    t = np.arange(len(X_test))# 创建t变量
    plt.plot(t, y_test, 'r', linewidth=1, marker='.', label='real') # 绘制y_test曲线
    plt.plot(t, y_predict, 'g', linewidth=1, marker='.', label='predict') # 绘制predict曲线
    plt.legend()# 设置图例
    plt.savefig('./MPS/Result/test.png')
    plt.show()
    return df


# %%
def preHandleDat(df,isTrain=True):
    from sklearn.preprocessing import StandardScaler #平均&變異數標準化 平均值為0，方差為1。
    from sklearn.preprocessing import MinMaxScaler #最小最大值標準化[0,1]
    from sklearn.preprocessing import RobustScaler #中位數和四分位數標準化
    from sklearn.preprocessing import MaxAbsScaler #絕對值最大標準化

    
    #=================
    #刪除不必要的欄位
    #=================
    drop_cols=['MFG_DATE','ARRIVAL_WIP_QTY','NO_HOLD_QTY']
    df = df.drop(drop_cols, axis=1)
    
    #========================
    # 缺漏值填空
    #========================
    # df = df.fillna(df.median())
    df = df.fillna(method='bfill')
    # df = df.fillna(df.mean())        
    #==================================================
    #1.特徵縮放
    #==================================================
    
    #特徵縮放欄位 List(排除Target TRCT)==================
    # num_cols=['M_NUM','UP_TIME','C_UP_TIME','LOT_SIZE','C_LOT_SIZE','EQP_UTIL','C_EQP_UTIL','U','PROCESS_TIME','WIP_QTY','NO_HOLD_QTY','MOVE_QTY','ARRIVAL_WIP_QTY','RUN_WIP_RATIO','CLOSE_WIP_QTY','MANAGEMENT_WIP_QTY','C_TC','C_CLOSE_WIP','C_TOOLG_LOADING','C_TOOL_LOADING','DISPATCHING','HOLD_RATE','ENG_LOT_RATE','HOT_LOT_RATE','[BACKUP]','REWORK_RATE','QUE_LOT_RATE','SAMPLING_RATE','NUM_RECIPE','CHANGE_RECIPE','BATCH_SIZE']

    num_cols=['M_NUM','UP_TIME','C_UP_TIME','LOT_SIZE','C_LOT_SIZE','EQP_UTIL','C_EQP_UTIL','U','PROCESS_TIME','WIP_QTY','RUN_WIP_RATIO','C_TC','HOLD_RATE','ENG_LOT_RATE','HOT_LOT_RATE','QUE_LOT_RATE'] #'ARRIVAL_WIP_QTY',
    df_train_scal = df
    #rescaling 特徵縮放 StandardScaler-------------------------------------
    # std_scaler = StandardScaler()
    # std_scaler.fit(df_train[num_cols])
    # df_train_scal[num_cols]= std_scaler.transform(df_train_scal[num_cols])


    #rescaling 特徵縮放 MinMaxScaler-------------------------------------
    minMax_scaler = MinMaxScaler()
    minMax_scaler.fit(df[num_cols])
    df_train_scal[num_cols]= minMax_scaler.transform(df_train_scal[num_cols])

    #rescaling 特徵縮放 RobustScaler-------------------------------------
    # robust_scaler = RobustScaler()
    # robust_scaler.fit(df_train[num_cols])
    # df_train_scal[num_cols] = robust_scaler.transform(df_train_scal[num_cols])
    
    #==================================================
    #2.one hot encoder
    #==================================================

    target_cols=['MOVE_QTY']
    cat_cols = ['TOOLG_ID']
    global df2_train_eh_before
    df_train_eh =pd.get_dummies(df_train_scal.drop(target_cols, axis=1),columns=cat_cols)
    if isTrain:
        df2_train_eh_before = df_train_eh
    else:
        df_train_eh = df_train_eh.reindex(columns = df2_train_eh_before.columns, fill_value=0)


    X_dropped = np.asarray(df_train_eh)

    Y_dropped = np.asarray(df[target_cols]) #'MOVE_QTY'
    return X_dropped,Y_dropped
    


# %%
def readDataFromFile(file_path):
    df = pd.read_csv(file_path)
    return df

# %%
def EDA(df_train,targetfeat='MOVE_QTY'):
    
    pairwise_analysis='off' #相關性和其他型別的資料關聯可能需要花費較長時間。如果超過了某個閾值，就需要設定這個引數為on或者off，以判斷是否需要分析資料相關性。
    report_train = sv.analyze([df_train, 'train'],
                                    target_feat= targetfeat
    )
    report_train.show_html(filepath='./MPS/sweetvizHTML/train_report.html' ) # 儲存為html的格式

    # compare_subsets_report = sv.compare_intra(df_train,
    #                                         df_train['Finish']==1, # 給條件區分
    #                                         ['Finish', 'notFinish'], # 為兩個子資料集命名 
    #                                         target_feat='NO_HOLD_QTY',
    #                                         )

    # compare_subsets_report.show_html(filepath='./sweetvizHTML/HW2_Compare_report.html')


# %%
df_train_orign=readDataFromFile('./MPS/data/TRCT_TrainingData_20210131.csv')
#df_train_orign['MFG_DATE'] = pd.to_datetime(test['MFG_DATE'],format='%Y%m%d')  ##不知道為什麼直接轉日期有些會不正確
df_train_orign['MFG_DATE'] = df_train_orign['MFG_DATE'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))

df_train = df_train_orign
# df_train.info()
# 1. 查看缺失情况
# print(df_train.isnull().sum())
# print(df_train.describe())# 128683

#df2_test = df2_test.dropna()# 刪除null值

# 'MFG_DATE','TOOLG_ID','IS_HOLIDAY','M_NUM','UP_TIME','C_UP_TIME','LOT_SIZE','C_LOT_SIZE','EQP_UTIL','C_EQP_UTIL','U','PROCESS_TIME','WIP_QTY','NO_HOLD_QTY','MOVE_QTY','ARRIVAL_WIP_QTY','RUN_WIP_RATIO','CLOSE_WIP_QTY','MANAGEMENT_WIP_QTY','C_TC','C_CLOSE_WIP','C_TOOLG_LOADING','C_TOOL_LOADING','DISPATCHING','HOLD_RATE','ENG_LOT_RATE','HOT_LOT_RATE','[BACKUP]','REWORK_RATE','QUE_LOT_RATE','SAMPLING_RATE','NUM_RECIPE','CHANGE_RECIPE','BATCH_SIZE'
# num_cols=['WIP_QTY','NO_HOLD_QTY','M','U','PT','UP_TIME','EQP_UTIL','TC','CS','AI','C_AI']
# missData(df_train)

#刪除columns 值是空的()
df_train = df_train.dropna(axis=1, how='all')
 
# #刪除rows ,target值是空的()
df_train = df_train[df_train['MOVE_QTY'].notna()]
df_train = df_train[df_train['NO_HOLD_QTY'].notna()]
# df_train.info()
print(df_train.isnull().sum())


# %%


# %% [markdown]
# ## 資料分析 Tool

# %%

#資料分析 Tool
#EDA(df_train,'TRCT')



# %%
# 檢查資料處理  value  是不是有 無限大
# print(np.all(np.isfinite(x)))
# print(np.all(np.isfinite(y)))

# %% [markdown]
# # 模型 訓練
df_train1 = df_train[df_train['MFG_DATE'] <  pd.to_datetime(final_date)]
#df_train1['MFG_DATE'].max()


# %%

trainLR(df_train1)

# %% [markdown]
# # 推估當天 28天 df_sum28
test = df_train1  
df = test.groupby('TOOLG_ID').apply(lambda x: x.set_index('MFG_DATE').resample('1D').first())

# %%
num_cols=['M_NUM','UP_TIME','C_UP_TIME','LOT_SIZE','C_LOT_SIZE','EQP_UTIL','C_EQP_UTIL','U','PROCESS_TIME','WIP_QTY','NO_HOLD_QTY', 'ARRIVAL_WIP_QTY','RUN_WIP_RATIO','C_TC','HOLD_RATE','ENG_LOT_RATE','HOT_LOT_RATE','QUE_LOT_RATE','MOVE_QTY']
df_sum28 = df.groupby(level=0)[num_cols].apply(lambda x: x.shift().rolling(min_periods=1,window=28).mean()).reset_index()


# %%
#df_sum28['MFG_DATE'].max()

# %% [markdown]
# ## 抓最後一天的數據  來預測當天的值 df_test_today
# 

# %%
df_test_today=df_sum28.loc[df_sum28['MFG_DATE']==df_sum28['MFG_DATE'].max()]

# %%

df_test_today['MFG_DATE'] = df_sum28['MFG_DATE'].max()+ datetime.timedelta(days=1)
weekno = df_test_today['MFG_DATE'].max().weekday()

if weekno < 5:
   df_test_today['IS_HOLIDAY'] = 1.0527
else:  # 5 Sat, 6 Sun
   df_test_today['IS_HOLIDAY'] =1

# %% [markdown]
# ##取得當天的生產資訊
# #### C_LOT_SIZE
# #### LOT_SIZE
# #### PROCESS_TIME
# #### WIP_QTY
# #### NO_HOLD_QTY
# #### ARRIVAL_WIP_QTY
# #### HOLD_RATE
# #### ENG_LOT_RATE
# #### HOT_LOT_RATE
# #### QUE_LOT_RATE

# %%


real_data_cols_withkeys =['MFG_DATE','TOOLG_ID','C_LOT_SIZE','LOT_SIZE','PROCESS_TIME','WIP_QTY','NO_HOLD_QTY','ARRIVAL_WIP_QTY','HOLD_RATE','ENG_LOT_RATE','HOT_LOT_RATE','QUE_LOT_RATE','MOVE_QTY']
real_data_cols =['C_LOT_SIZE','LOT_SIZE','PROCESS_TIME','WIP_QTY','NO_HOLD_QTY','ARRIVAL_WIP_QTY','HOLD_RATE','ENG_LOT_RATE','HOT_LOT_RATE','QUE_LOT_RATE','MOVE_QTY']
df_map_today = df_train[real_data_cols_withkeys] 
df_map_today=df_map_today.loc[df_map_today['MFG_DATE']==df_test_today['MFG_DATE'].max()]

for index, row in df_test_today.iterrows():
    r = df_map_today[real_data_cols].loc[(df_map_today['MFG_DATE']==row['MFG_DATE'])  & (df_map_today['TOOLG_ID']==row['TOOLG_ID'])]
    if(r.any(axis=None)):
        for col in real_data_cols:
            df_test_today.loc[index,col]  =r[col].values
    else:
         df_test_today.drop(index, inplace=True)
print(df_train1.shape)

df_test_today.to_csv('./MPS/data/MyToday20200120_CT.csv')
testLR(df_test_today).to_csv('./MPS/data/MyToday20200120_result_CT.csv')



