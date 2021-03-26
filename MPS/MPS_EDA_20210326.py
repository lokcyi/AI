 
# %%
import tensorflow as tf
import numpy as np 
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn import metrics
from pickle import dump
from pickle import load
import joblib
import re
import sweetviz as sv
import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from xgboost import plot_importance 
final_date = '2021-01-18'
target_ToolGID = 'PK_DUVKrF'
# target_ToolGID = 'XE_Sorter'
# target_ToolGID = 'MA_Alps'
# target_ToolGID = 'PW_PIX'


# %%
#未來所有toolg_list 一起做one hot encoder
def getDummyToolG():
    cat_cols = ['TOOLG_ID']
    df_toolgList_orign=readDataFromFile('./data/ToolG_List.csv')
    df_train_eh =pd.get_dummies(df_toolgList_orign,columns=cat_cols)
   
    return df_train_eh.columns
   


# %%
#刪除outlier
def iqrfilter(df, colname, bounds = [.25, .75]):
    s = df[colname]
    Q1 = df[colname].quantile(bounds[0])
    Q3 = df[colname].quantile(bounds[1])
    IQR = Q3 - Q1
    # print(IQR,Q1,Q3,Q1 - 1.5*IQR,Q3+ 1.5 * IQR)
    if bounds[0]==0:
        return df[~s.clip(*[Q1,Q3+ 1.5 * IQR]).isin([Q1,Q3+ 1.5 * IQR])]
    else:
        return df[~s.clip(*[Q1 - 1.5*IQR,Q3+ 1.5 * IQR]).isin([Q1 - 1.5*IQR,Q3+ 1.5 * IQR])]


# %%
#NN tunning 參照 Function
def build_model(hp):
    
    # from tensorflow.keras import layers
    from kerastuner.tuners import RandomSearch
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=hp.Int('units',
                                        min_value=32,
                                        max_value=512,
                                        step=32),
                           activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice('learning_rate',
                      values=[1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model

# %% [markdown]
# # 繪製模型預測圖

# %%
def drawModelResult(modelType,TOOLG_ID,x,y_actual,y_predict28 , y_predict_actrual,imagepath):

    plt.figure(figsize=(6, 5))# 设置图片尺寸
    plt.title('TOOLG_ID:'+ TOOLG_ID+ " ("+modelType+")")
    # t = np.arange(len(X_dropped))# 创建t变量
    plt.plot(x, y_actual, 'b', linewidth=1, marker='.', label='actual') # 绘制y_test曲线
    plt.plot(x, y_predict28, 'r', linewidth=1, marker='.', label='predict on 28 days') # 绘制predict曲线
    plt.plot(x, y_predict_actrual, 'k', linewidth=1, marker='.', label='predict on actual') # 绘制predict曲线
    plt.legend()
        # target_ToolGID = 'PK_DUVKrF'
    # target_ToolGID = 'XE_Sorter'
    # target_ToolGID = 'MA_Alps'
    # target_ToolGID = 'PW_PIX'
    if TOOLG_ID=='PK_DUVKrF' :
        plt.yticks(np.linspace(0.15,0.45,9))
    elif TOOLG_ID=='XE_Sorter' :
        plt.yticks(np.linspace(0.0,0.12,9))
    elif TOOLG_ID=='MA_Alps' :
        plt.yticks(np.linspace(0.0,0.5,9))
    elif TOOLG_ID=='PW_PIX' :
        plt.yticks(np.linspace(0.0,0.1,9))    
    plt.savefig(imagepath)

# %% [markdown]
# ## NN 模型

# %%
#NN tunning Function
def tuningNN(load_model,x,y):
    #tf.keras.wrappers.scikit_learn.KerasClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import make_scorer
    from sklearn.metrics import accuracy_score, precision_score, recall_score,r2_score
    
    build_model = lambda: load_model
    Kmodel = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=build_model, verbose=1)
    scorers = {
        # 'precision_score': make_scorer(precision_score),
        # 'recall_score': make_scorer(recall_score),
        # 'accuracy_score': make_scorer(accuracy_score)
        'r2_score':make_scorer(r2_score)
        }
 
    distributions = dict(batch_size = [ 16,32,64,75], epochs = [50, 75,100] #  , optimizer=['rmsprop', 'adam']#hidden_layers=[[64], [32]]
    )
 
    clf = RandomizedSearchCV(Kmodel, distributions, scoring=scorers,random_state=0,n_iter = 5, cv = 2, verbose=10,refit='r2_score')

    from joblib import Parallel, delayed, parallel_backend
    with parallel_backend('threading',n_jobs=12):
        search = clf.fit(x, y)
    
    print(clf.best_estimator_.model) 
    # 評估，打分數
    print(f"最佳準確率: {clf.best_score_}，最佳參數組合：{clf.best_params_}")
 
    return clf.best_estimator_.model



# %%
def trainNN(df):
    import tensorflow as tf
    save_model_tool = './NN/training_model2.h5'
    # save_model_tool = getSavePath(df['TOOLG_ID'].iloc[0],save_model)
    df_result = df.copy(deep=False)
    X_dropped, Y_dropped = preHandleDat(df_result,True)
    # tuneNN(X_dropped,Y_dropped)
    
    # 拆分train validation set NN fit 可以自己拆
    # X_train, X_test,y_train,y_test = train_test_split(X_dropped,Y_dropped,test_size =0.1,random_state=587)
    
    #1.建立模型(Model)
    #將Layer放入Model中
    # Activation Functions
    # A.softmax：值介於 [0,1] 之間，且機率總和等於 1，適合多分類使用。
    # B.sigmoid：值介於 [0,1] 之間，且分布兩極化，大部分不是 0，就是 1，適合二分法。
    # C.Relu (Rectified Linear Units)：忽略負值，介於 [0,∞] 之間。
    # D.tanh：與sigmoid類似，但值介於[-1,1]之間，即傳導有負值。
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=16,input_shape=[X_dropped.shape[1]]), # units：输出维度
        tf.keras.layers.Dense(units=16, kernel_initializer='random_uniform', activation='relu'),
        tf.keras.layers.Dense(units=1, kernel_initializer='random_uniform',activation='relu')      
        ])
    #model.summary()
    #======================================================================================
    # 定義 tensorboard callback
    tensorboard_callback = [tf.keras.callbacks.TensorBoard(log_dir='D:/Projects/AI/POC/homework/logs2')]
    
    #========================================================================
    # SGD
    sgd = tf.keras.optimizers.SGD(lr=0.20, momentum=0.0, decay=0.0, nesterov=False)
    # 随机梯度下降优化器。
    # 包含扩展功能的支持： - 动量（momentum）优化, - 学习率衰减（每次参数更新后） - Nestrov 动量 (NAG) 优化
    # 参数
    # lr: float >= 0. 学习率。
    # momentum: float >= 0. 参数，用于加速 SGD 在相关方向上前进，并抑制震荡。
    # decay: float >= 0. 每次参数更新后学习率衰减值。
    # nesterov: boolean. 是否使用 Nesterov 动量。
    #========================================================================
  
    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(lr=0.01) #, optimizer=tf.keras.optimizers.SGD(lr=0.2)
                        , metrics = [ 'mae', 'mape'])   
    #需要tunning 時開啟
    #model = tuningNN(model,X_dropped,Y_dropped)                                            
    #======================================================================================
    #3.訓練 fit：以compile函數進行訓練，指定訓練的樣本資料(x, y)，並撥一部分資料作驗證，還有要訓練幾個週期、訓練資料的抽樣方式。
    train_history = model.fit(x=X_dropped, y=Y_dropped,
                validation_split=0.10, epochs=75, batch_size=50, verbose=0) #,shuffle=True validation_split=0.1, 用最後的10%資料驗證 batch_size=200: 每一批次200筆資料
    model.save(save_model_tool)

    print(train_history.history.keys())
    #dict_keys(['loss', 'accuracy', 'mse', 'mae', 'mape', 'val_loss', 'val_accuracy', 'val_mse', 'val_mae', 'val_mape'])
    
    # 當RMSE收斂至接近0.02，且MAPE接近10%，即完成模型之訓練
    figure, axis_1 = plt.subplots()
    plt.title(df['TOOLG_ID'].iloc[0]) # title
    plt.xlabel('Epoch Number')
    plt.ylabel("Loss Magnitude")

    loss = axis_1.plot(train_history.history['loss'], label = 'loss')

    axis_2 = axis_1.twinx()
    # mse = axis_2.plot(train_history.history['mse'], label = 'mse',color='red' ) 
    mse = axis_2.plot(train_history.history['mape'], label = 'mape',color='red' ) 
    
    # mape = axis_2.plot(train_history.history['mape'], label = 'mape' )# 準確度 接近10%
        
    axis_1.legend(loc='upper left',fontsize='large')
    axis_2.legend(loc='upper right',fontsize='large')
    plt.show()


# %%
def testNN(df,df_real):
    import tensorflow as tf
    save_model_tool = './NN/training_model2.h5'
    model = tf.keras.models.load_model(save_model_tool)
    df_result=df.copy(deep=True)
    
    X_dropped,Y_dropped = preHandleDat(df,False)
    y_predict = model.predict(X_dropped)
    df_result['predict'] = y_predict # 預測
    

    df_result_real=df_real.copy(deep=True)
    X_dropped_real,Y_dropped_real = preHandleDat(df_result_real,False)
    y_predict_real = model.predict(X_dropped_real)
    df_result_real['predict'] = y_predict_real # 預測

    drawModelResult('NN',df['TOOLG_ID'].iloc[0],df_result['MFG_DATE'],Y_dropped, y_predict,y_predict_real,'./Result/NNtest2.png')

    return df_result


# %%

# ## XG 模型

def trainXG(df):
    import xgboost as xgb
    
    import joblib

    X,Y = preHandleDat(df,True)
    
    #拆分train validation set
    X_train, X_test,y_train,y_test = train_test_split(X,Y,test_size =0.2,random_state=587)
    cv_params = {'n_estimators': [300,400,500,600],'max_depth':[7,11,13,15,17],'min_child_weight':[1,3,5,7,9]}
    other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
    model = xgb.XGBRegressor(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)

    optimized_GBM.fit(X_train, y_train) 

    test_score = optimized_GBM.score(X_test,y_test)

    print('test 得分:{0}'.format(test_score))
    # evalute_result = optimized_GBM.grid_scores_
    # print('每輪迭代執行結果:{0}'.format(evalute_result))
    
    print('引數的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
    print('cv_results_',optimized_GBM.cv_results_)

    bst_model = optimized_GBM.estimator
    print(bst_model)

    
    bst_model.fit(X_train, y_train)
    plot_importance(bst_model)#,max_num_features=10)
    plt.show()

    joblib.dump(bst_model, 'XG_model')


# %%
def testXG(df,df_real):
    X_test,y_test = preHandleDat(df,False)
    # print(X_test)
    #print(X_test,y_test)
    df_result=df.copy(deep=True)
    loaded_model = joblib.load('XG_model')
    y_predict = loaded_model.predict(X_test)
    df_result['predict'] = y_predict
    
    loaded_model.score(X_test, y_test)
    print("r2:", loaded_model.score(X_test, y_test))
    
    df_result_real=df_real.copy(deep=True)
    X_dropped_real,Y_dropped_real = preHandleDat(df_result_real,False)
    y_predict_real = loaded_model.predict(X_dropped_real)
    df_result_real['predict'] = y_predict_real # 預測

    drawModelResult('XG',df['TOOLG_ID'].iloc[0],df_result['MFG_DATE'],y_test, y_predict,y_predict_real,'./Result/XGtest2.png')

    
    return df_result


# ## LR 模型

# %%
def trainLR(df):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    import joblib

    X,Y = preHandleDat(df,True)
    
    #拆分train validation set
    X_train, X_test,y_train,y_test = train_test_split(X,Y,test_size =0.01,random_state=587)
    
    model = LinearRegression(fit_intercept=False, normalize=False, copy_X=False) #fit_intercept=False
   
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    
    #print(y_predict)
    print("r2:", model.score(X_test, y_test))# 拟合优度R2的输出方法
    print("MAE:", metrics.mean_absolute_error(y_test, y_predict))# 用Scikit_learn计算MAE
    print("MSE:", metrics.mean_squared_error(y_test, y_predict)) # 用Scikit_learn计算MSE
    print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_predict)))# 用Scikit_learn计算RMSE
    print("intercept_ :",model.intercept_)
    
    joblib.dump(model, 'LR_model')
    

    return y_predict


# %%
def testLR(df,df_real):
    from sklearn.linear_model import LinearRegression
    
    X_test,y_test = preHandleDat(df,False)
    # print(X_test)
    #print(X_test,y_test)
    loaded_model = joblib.load('LR_model')
    y_predict = loaded_model.predict(X_test)
    df_result=df.copy(deep=True)
    df_result['predict'] = y_predict
    
    loaded_model.score(X_test, y_test)
    print("r2:", loaded_model.score(X_test, y_test))
    
    df_result_real=df_real.copy(deep=True)
    X_dropped_real,Y_dropped_real = preHandleDat(df_result_real,False)
    y_predict_real = loaded_model.predict(X_dropped_real)
    df_result_real['predict'] = y_predict_real # 預測

    drawModelResult('LR',df['TOOLG_ID'].iloc[0],df_result['MFG_DATE'],y_test, y_predict,y_predict_real,'./Result/LR.png')

    return df_result


# # 資料 預處理 preHandleDat()
def preHandleDat(df,isTrain=True):
    from sklearn.preprocessing import StandardScaler #平均&變異數標準化 平均值為0，方差為1。
    from sklearn.preprocessing import MinMaxScaler #最小最大值標準化[0,1]
    from sklearn.preprocessing import RobustScaler #中位數和四分位數標準化
    from sklearn.preprocessing import MaxAbsScaler #絕對值最大標準化

    #=================
    #刪除不必要的欄位
    #=================


    # drop_cols=['MFG_DATE','ARRIVAL_WIP_QTY','MOVE_QTY','PROCESS_TIME','C_TC']
    # num_cols=['M_NUM','UP_TIME','C_UP_TIME','LOT_SIZE','C_LOT_SIZE','EQP_UTIL','C_EQP_UTIL','U','HOLD_RATE','ENG_LOT_RATE','HOT_LOT_RATE','QUE_LOT_RATE','NO_HOLD_QTY','WIP_QTY','IS_HOLIDAY','RUN_WIP_RATIO'] 
    drop_cols=['MFG_DATE','ARRIVAL_WIP_QTY','MOVE_QTY']
    num_cols=['M_NUM','UP_TIME','C_UP_TIME','LOT_SIZE','C_LOT_SIZE','EQP_UTIL','C_EQP_UTIL','U','PROCESS_TIME','C_TC','HOLD_RATE','ENG_LOT_RATE','HOT_LOT_RATE','QUE_LOT_RATE','NO_HOLD_QTY','WIP_QTY','IS_HOLIDAY','RUN_WIP_RATIO']

    target_cols=['TRCT']
    cat_cols = ['TOOLG_ID']
    df = df.drop(drop_cols, axis=1)
   
    #========================
    # 缺漏值填空
    #========================
    # df = df.fillna(df.median())
    #df = df.fillna(method='bfill') #往後
    df = df.fillna(method='ffill') #往前
    # df = df.fillna(df.mean())  #用平均值取代 nan   
    # df = df.dropna() # 刪除null值   
    #df['ColA'].fillna(value=0, inplace=True) #用 0 取代 nan
    #df['ColA'].fillna(value=df.groupby('ColB')['ColA'].transform('mean'), inplace=True) #利用 groupby()同一group 的平均值

    #==================================================
    #1.特徵縮放
    #==================================================
    
    #特徵縮放欄位 List(排除Target TRCT)==================
    # num_cols=['M_NUM','UP_TIME','C_UP_TIME','LOT_SIZE','C_LOT_SIZE','EQP_UTIL','C_EQP_UTIL','U','PROCESS_TIME','WIP_QTY','NO_HOLD_QTY','MOVE_QTY','ARRIVAL_WIP_QTY','RUN_WIP_RATIO','CLOSE_WIP_QTY','MANAGEMENT_WIP_QTY','C_TC','C_CLOSE_WIP','C_TOOLG_LOADING','C_TOOL_LOADING','DISPATCHING','HOLD_RATE','ENG_LOT_RATE','HOT_LOT_RATE','[BACKUP]','REWORK_RATE','QUE_LOT_RATE','SAMPLING_RATE','NUM_RECIPE','CHANGE_RECIPE','BATCH_SIZE']

   #'ARRIVAL_WIP_QTY',,'WIP_QTY'
    df_train_scal = df.copy(deep=False)
    global df_cols
    df_cols = df_train_scal.columns
    
    if isTrain:
        
        
        #rescaling 特徵縮放 StandardScaler-------------------------------------
        # scaler = StandardScaler()
        # scaler.fit(df_train[num_cols])
        # df_train_scal[num_cols]= scaler.transform(df_train_scal[num_cols])
    
        #rescaling 特徵縮放 MinMaxScaler-------------------------------------
        scaler = MinMaxScaler()
        scaler.fit(df[num_cols])
        df_train_scal[num_cols]= scaler.transform(df_train_scal[num_cols])

        #rescaling 特徵縮放 RobustScaler-------------------------------------
        # scaler = RobustScaler()
        # scaler.fit(df_train[num_cols])
        # df_train_scal[num_cols] = scaler.transform(df_train_scal[num_cols])
        
       
        dump(scaler, open('scaler.pkl', 'wb'))
        # # save the scaler
        # dump(scaler, open('scaler.pkl', 'wb'))
    else:
        # load the scaler
        scaler = load(open('scaler.pkl', 'rb'))
        df_train_scal[num_cols] = scaler.transform(df[num_cols])


    df_train_scal.to_csv('./data/df_train_scal.csv')
    #==================================================
    #2.one hot encoder
    #==================================================
    # target_cols=['MOVE_QTY']
 
    # global df2_train_eh_before
    df_train_eh =pd.get_dummies(df_train_scal.drop(target_cols, axis=1),columns=cat_cols)
    df_train_eh.to_csv('./data/df2_train_eh_before.csv')

    
    
    # # Get missing columns in the training test
    # eh_cols =  getDummyToolG()
    # missing_cols = set( eh_cols ) - set( df_train_eh.columns )

    # # Add a missing column in test set with default value equal to 0
    # for c in missing_cols:
    #     df_train_eh[c] = 0
   
    
    if isTrain:
        df2_train_eh_before = df_train_eh.copy(deep=False)
        df2_train_eh_before.head(0).to_csv('./data/train_eh.csv',index=0) #不保存行索引
    else:
        df2_train_eh_before=readDataFromFile('./data/train_eh.csv')
        df_train_eh = df_train_eh.reindex(columns = df2_train_eh_before.columns, fill_value=0)
        # Ensure the order of column in the test set is in the same order than in train set
        df_train_eh = df_train_eh[df2_train_eh_before.columns]  
 

    df_train_eh.to_csv('./data/df_train_eh.csv')
           
    X_dropped = np.asarray(df_train_eh)

    Y_dropped = np.asarray(df[target_cols]) 
    return X_dropped,Y_dropped
    


# %%
def readDataFromFile(file_path):
    _df = pd.read_csv(file_path)
    return _df


# %%
def EDA(df_train,targetfeat='MOVE_QTY'):
    
    pairwise_analysis='off' #相關性和其他型別的資料關聯可能需要花費較長時間。如果超過了某個閾值，就需要設定這個引數為on或者off，以判斷是否需要分析資料相關性。
    report_train = sv.analyze([df_train, 'train'],
                                    target_feat= targetfeat,
                                    pairwise_analysis = pairwise_analysis
    )
    report_train.show_html(filepath='./sweetvizHTML/train_report.html' ) # 儲存為html的格式

    # compare_subsets_report = sv.compare_intra(df_train,
    #                                         df_train['Finish']==1, # 給條件區分
    #                                         ['Finish', 'notFinish'], # 為兩個子資料集命名 
    #                                         target_feat='NO_HOLD_QTY',
    #                                         )

    # compare_subsets_report.show_html(filepath='./sweetvizHTML/HW2_Compare_report.html')


# %%
df_train_orign=readDataFromFile('./data/TRCT_TrainingData_20210131.csv')
 
df_train_orign = df_train_orign.loc[df_train_orign['TOOLG_ID']==target_ToolGID]
#df_train_orign['MFG_DATE'] = pd.to_datetime(test['MFG_DATE'],format='%Y%m%d') 
df_train_orign['MFG_DATE'] = df_train_orign['MFG_DATE'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))

df_train = df_train_orign

df_train =iqrfilter(df_train_orign,'M_NUM',[0.25, 1]) 


# df_train.info()
# 1. 查看缺失情况
# print(df_train.isnull().sum())
# print(df_train.describe())# 128683

#刪除columns 值是空的()
df_train = df_train.dropna(axis=1, how='all')
 
# #刪除rows ,target值是空的()
df_train = df_train[df_train['MOVE_QTY'].notna()]
df_train = df_train[df_train['NO_HOLD_QTY'].notna()]
# df_train['TRCT']= df_train['NO_HOLD_QTY']/df_train['MOVE_QTY']
df_train['TRCT']= df_train['NO_HOLD_QTY']/df_train['MOVE_QTY']

df_train.info()

df_train.isnull().sum()
 
# 檢查資料處理  value  是不是有 無限大
# x,y=preHandleDat(df_train)
 
# # # np.isnan(y.any()) #and gets False
# # # np.isfinite(y.all()) #and gets True

# print(np.all(np.isfinite(x)))
# print(np.all(np.isfinite(y)))

 

 
# ## 資料分析 Tool
# EDA(df_train,'TRCT')


# %%
df_train1 = df_train[df_train['MFG_DATE'] <  pd.to_datetime(final_date)]

# df_train1 =iqrfilter(df_train1,'M_NUM',[0.25, 1]) 
df_train1['MFG_DATE'].max()

# # 模型 訓練
# %%
def getsum28(df_train,final_date):
    
    test = df_train[df_train['MFG_DATE'] <  pd.to_datetime(final_date)]
    # test['MFG_DATE'] = pd.to_datetime(test['MFG_DATE'],format='%Y%m%d') 
    print(test['MFG_DATE'].max())
    df = test.groupby('TOOLG_ID').apply(lambda x: x.set_index('MFG_DATE').resample('1D').first())

    num_cols=['M_NUM','UP_TIME','C_UP_TIME','LOT_SIZE','C_LOT_SIZE','EQP_UTIL','C_EQP_UTIL','U','PROCESS_TIME','WIP_QTY','NO_HOLD_QTY', 'ARRIVAL_WIP_QTY','RUN_WIP_RATIO','C_TC','HOLD_RATE','ENG_LOT_RATE','HOT_LOT_RATE','QUE_LOT_RATE','MOVE_QTY']
    
    df_sum28 = df.groupby(level=0)[num_cols].apply(lambda x: x.shift().rolling(min_periods=1,window=28).mean()).reset_index()
    
    #df_train1['MFG_DATE'].max()
    df_sum28['MFG_DATE'].max()
    #抓最後一天的數據 來預測當天的值 df_test_today
    df_test_today=df_sum28.loc[df_sum28['MFG_DATE']==df_sum28['MFG_DATE'].max()]
     
    # df_test_today['MFG_DATE'] = df_sum28['MFG_DATE'].max()+ datetime.timedelta(days=1)
    weekno = df_test_today['MFG_DATE'].max().weekday()

    if weekno < 5:
        df_test_today['IS_HOLIDAY'] = 1
    else:  # 5 Sat, 6 Sun
        df_test_today['IS_HOLIDAY'] =1.0527

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
        
    #測試集的答案 驗證用
    df_test_today['TRCT']= df_test_today['NO_HOLD_QTY']/df_test_today['MOVE_QTY']

    return df_test_today


# %%
# 訓練資料集 是用28天的平均(不需要使用了)
def generationTraininDatFrame(_dfRaw):
     
    test = _dfRaw
   
    df = test.groupby('TOOLG_ID').apply(lambda x: x.set_index('MFG_DATE').resample('1D').first())
    
    num_cols=['M_NUM','UP_TIME','C_UP_TIME','LOT_SIZE','C_LOT_SIZE','EQP_UTIL','C_EQP_UTIL','U','PROCESS_TIME','WIP_QTY','NO_HOLD_QTY', 'ARRIVAL_WIP_QTY','RUN_WIP_RATIO','C_TC','HOLD_RATE','ENG_LOT_RATE','HOT_LOT_RATE','QUE_LOT_RATE','MOVE_QTY']
    
    df_sum28 = df.groupby(level=0)[num_cols].apply(lambda x: x.shift().rolling(min_periods=1,window=28).mean()).reset_index()
    
    _df_result = pd.DataFrame(columns = df_sum28.columns)
 
    for index, df_sum28_row  in df_sum28.iterrows():
        
        if index<=28:
            continue
      
        df_test_today=df_sum28_row
      
        
        real_data_cols_withkeys =['MFG_DATE','TOOLG_ID','C_LOT_SIZE','LOT_SIZE','PROCESS_TIME','WIP_QTY','NO_HOLD_QTY','ARRIVAL_WIP_QTY','HOLD_RATE','ENG_LOT_RATE','HOT_LOT_RATE','QUE_LOT_RATE','MOVE_QTY','IS_HOLIDAY']
        real_data_cols =['C_LOT_SIZE','LOT_SIZE','PROCESS_TIME','WIP_QTY','NO_HOLD_QTY','ARRIVAL_WIP_QTY','HOLD_RATE','ENG_LOT_RATE','HOT_LOT_RATE','QUE_LOT_RATE','MOVE_QTY','IS_HOLIDAY']

        #取得當天的WIP 實際資料===========================
        df_map_today = _dfRaw[real_data_cols_withkeys] 
        df_map_today=df_map_today.loc[df_map_today['MFG_DATE']==df_sum28_row['MFG_DATE']]
        df_map_today=df_map_today.loc[df_map_today['TOOLG_ID']==df_sum28_row['TOOLG_ID']] #+ datetime.timedelta(days=1)]
 
        if df_map_today.shape[0]==0:
            continue
        for col in real_data_cols:
            df_test_today[col] = df_map_today.iloc[0][col] 
            
        #測試集的答案 驗證用
        df_test_today['TRCT']= df_test_today['NO_HOLD_QTY']/df_test_today['MOVE_QTY']
        _df_result = _df_result.append(df_test_today,ignore_index=True)
        
    return _df_result
# ## 訓練集取28天平均 df_train_sum28 
# ### df_train_sum28_1 <==區分訓練與驗證
df_train_sum28_org = generationTraininDatFrame(df_train)
print(df_train_sum28_org)
df_train_sum28 = df_train_sum28_org


# %%
plt.title('TOOLG_ID:'+ df_train['TOOLG_ID'].iloc[0])

plt.ylabel("M_NUM")
plt.xlabel("date")
plt.plot(df_train['MFG_DATE'] , df_train['M_NUM'])
# plt.savefig('./'+df['TOOLG_ID'].iloc[0]+'.pdf',width=600, height=350)
plt.show()


# %%


#check偏離值(M_NUM)

plt.title('TOOLG_ID:'+ df_train['TOOLG_ID'].iloc[0])
plt.ylabel("M_NUM")
plt.xlabel("date")
plt.plot(df_train_sum28['MFG_DATE'] , df_train_sum28['M_NUM'])
# plt.savefig('./'+df['TOOLG_ID'].iloc[0]+'.pdf',width=600, height=350)
plt.show()

df_train_sum28_1  =iqrfilter(df_train_sum28,'TRCT',[.25, 1]) 
df_train_sum28_1 = df_train_sum28[df_train_sum28['MFG_DATE'] <  pd.to_datetime(final_date)]
df_train_sum28.to_csv('./data/df_train_sum28.csv')

# ## 計算測試集資料 取28天平均 df

# %%
df = pd.DataFrame(columns = df_train.columns)


for i in range(1,((df_train['MFG_DATE'].max()- datetime.datetime.strptime(final_date, "%Y-%m-%d")).days)+2 ):
    _final_date =   datetime.datetime.strptime(final_date, "%Y-%m-%d")+ datetime.timedelta(days=i)
 
    _df = getsum28(df_train,_final_date)
    df = df.append(_df,ignore_index=True)

# df.to_csv('./data/MyToday20200120_CT.csv')

# %% [markdown]
# ## 1. 28 均值 訓練 跑 LR
#實際值
df_test_real =  df_train[df_train['MFG_DATE'] >=  pd.to_datetime(final_date)]

#計算準確率
ef accsum(def_result):
    _accsum=0
    def_result[def_result['TRCT'] ==0.0]['TRCT']  =0.0001
    def_result[def_result['predict'] <0]['predict']  =0
    for index,row in def_result.iterrows():
        #避免當分母為0 會無法計算
        if row['TRCT'] <0 :
            row['TRCT']  =0.00001
        if row['TRCT'] ==0.0:
            row['TRCT']  =0.00001
        # print(row['TRCT'] )
        if 1- abs((row['predict'] - row['TRCT'])/row['TRCT'] ) >0 :
            
            _accsum+=(1- abs((row['predict'] - row['TRCT'])/row['TRCT'] ))
    
    return _accsum/def_result.shape[0]

trainLR(df_train_sum28_1)
def_result= testLR(df_train_sum28_1,df_train_sum28_1)
print("Train acc%:",accsum(def_result))
def_result= testLR(df,df_test_real)
print("Test acc%:",accsum(def_result)) 
def_result= testLR(df_test_real,df_test_real)
print("real acc%:",accsum(def_result))


# ## 2. 28 均值 訓練 跑 XG
trainXG(df_train_sum28_1)
def_result= testXG(df_train_sum28_1,df_train_sum28_1)
print("Train acc%:",accsum(def_result))
def_result= testXG(df,df_test_real)
print("Train acc%:",accsum(def_result))
def_result= testXG(df_test_real,df_test_real)
print("real acc%:",accsum(def_result))
 
# ## 3. 28 均值 訓練 跑 NN
 
trainNN(df_train_sum28_1)
def_result= testNN(df_train_sum28_1,df_train_sum28_1)
print("Train acc%:",accsum(def_result))
def_result= testNN(df,df_test_real)
print("test acc%:",accsum(def_result))
def_result= testNN(df_test_real,df_test_real)
print("real acc%:",accsum(def_result))

# # 訓練並測試模型( 每日歷史資料 訓練)
# ##  1. 每日歷史資料(df_train1) 跑 LR

trainLR(df_train1)
def_result = testLR(df_train1,df_train1)
print("train acc%:",accsum(def_result))
#計算準確率分數
def_result = testLR(df,df_test_real)
print("test acc%:",accsum(def_result))
#計算準確率分數
def_result= testLR(df_test_real,df_test_real)
print("real acc%:",accsum(def_result))

# ## 2.每日歷史資料 跑 XGBoost
#驗證訓練集 train
trainXG(df_train1)
def_result = testXG(df_train1,df_train1)
print("Train acc%:",accsum(def_result)) 
#驗證測試集  test
def_result = testXG(df,df_test_real)
print("Train acc%:",accsum(def_result)) 
def_result= testXG(df_test_real,df_test_real)
print("real acc%:",accsum(def_result))


# %%
df_train1.columns

# %% [markdown]
# ## 3.每日歷史資料 跑 NN
trainNN(df_train1)
def_result = testNN(df_train1,df_train1)
print("Train acc%:",accsum(def_result)) 
def_result = testNN(df,df_test_real)
print("test acc%:",accsum(def_result)) 
def_result= testNN(df_test_real,df_test_real)
print("real acc%:",accsum(def_result))

# df_test_real.to_csv("./data/df_test_real.csv")
# df_train_sum28_1.to_csv('./data/df_train_sum28_1.csv')
# df.to_csv('./data/df.csv')



