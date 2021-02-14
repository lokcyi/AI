import numpy as np  
import pandas as pd  


#讀取資料
# train_df = pd.read_csv(r'D:\PythonSample\Homework01\data20210128\Training_data_20210128.csv')
train_df = pd.read_csv('D:/projects/ai/poc/homework/training_data_20210128.csv')
train_df=train_df.head(1000)
#print(train_df)
#print(train_df.isnull().sum())
print("Total資料筆數:")
print(train_df.shape[0])

#移除異常值資料
train_df.loc[train_df['LOT_TYPE']=='FDY']
# train_df.loc[train_df['LAYER']!='XX']
print("只取LOT_TYPE=FDY 資料筆數:")
print(train_df.shape[0])

#遺漏值用平均值填補 (Sample)
#remain_mean = train_df['REMAIN_LAYER_SEQ'].mean()  
#train_df['REMAIN_LAYER_SEQ'] = train_df['REMAIN_LAYER_SEQ'].fillna(remain_mean)  
#train_df['REMAIN_LAYER_SEQ'] = train_df['REMAIN_LAYER_SEQ'].fillna(0)  
#print(train_df.isnull().sum())


#定義欄位屬性
train_df["DATA_DATE"] =train_df["DATA_DATE"].astype(str)
train_df["WS_DATE"] =train_df["WS_DATE"].astype(str)
train_df["ACTUAL_WP_OUT"] =train_df["ACTUAL_WP_OUT"].astype(str)
train_df["OP_NO"] =train_df["OP_NO"].astype(str)
# train_df["PRIORITY"] =train_df["PRIORITY"].astype(int))
train_df["PRIORITY"] =train_df["PRIORITY"].astype(str)



#新增欄位
#PROCESSDAYS(已處理天數) = DATA_DATE(資料快照日) -WS_DATE(投入日)
#NEEDDAYS(仍需N天才能產出) = ACTUAL_WP_OUT(實際產出日)-DATA_DATE
train_df=train_df.assign(PROCESSDAYS= ((pd.to_datetime(train_df["DATA_DATE"]) - pd.to_datetime(train_df["WS_DATE"]))/pd.Timedelta(1, 'D')).fillna(0).astype(int))
train_df=train_df.assign(NEEDDAYS= ((pd.to_datetime(train_df["ACTUAL_WP_OUT"]) - pd.to_datetime(train_df["DATA_DATE"]))/pd.Timedelta(1, 'D')).fillna(0).astype(int))

#移除不使用的column(欄)
# train_df = train_df.drop(columns=['DATA_DATE','IDX','LOT_ID','LOT_TYPE','WS_DATE','ACTUAL_WP_OUT']) 
train_df = train_df.drop(train_df.loc[:, '0E':'ZL'].columns, axis = 1)
train_df = train_df.drop(columns=['DATA_DATE','IDX','LOT_ID','LOT_TYPE','WS_DATE','LAYER','ACTUAL_WP_OUT'])
 
#print(train_df.head())

#刪除資料中有NaN的data
#print(train_df.isnull().sum())
train_df = train_df.dropna()

print("刪除NaN後資料筆數:")
print(train_df.shape[0])

#定義欄位屬性
# train_df["REMAIN_OP_SEQ"] =train_df["REMAIN_OP_SEQ"].astype(int)
# train_df["WIP_QTY"] =train_df["WIP_QTY"].astype(int)

# 使用get_dummies進行Category轉換    
train_df = pd.get_dummies(data=train_df, columns=['STATUS']) 
train_df = pd.get_dummies(data=train_df, columns=['CHIPNAME']) 
train_df = pd.get_dummies(data=train_df, columns=['OP_NO']) 
# train_df = pd.get_dummies(data=train_df, columns=['LAYER']) 
train_df = pd.get_dummies(data=train_df, columns=['PRIORITY'])
train_df = pd.get_dummies(data=train_df, columns=['IS_MAIN_ROUTE']) 
# #print(train_df.head())
# from sklearn.preprocessing import LabelEncoder
# labelencoder = LabelEncoder()
# train_df['PRIORITY'] = labelencoder.fit_transform(train_df['PRIORITY'])
# train_df['IS_MAIN_ROUTE'] = labelencoder.fit_transform(train_df['IS_MAIN_ROUTE'])
# train_df['STATUS'] = labelencoder.fit_transform(train_df['STATUS'])
# train_df['CHIPNAME'] = labelencoder.fit_transform(train_df['CHIPNAME'])


# 將 dataframe 轉換為 array
#ndarray = train_df.to_numpy()
#ndarray =np.asarray(train_df)
# ndarray =train_df.values  

# Separate labels with features 
Label =  np.asarray(train_df['NEEDDAYS'])
xdata = train_df.copy()
xdata = np.asarray(xdata.drop(columns=['NEEDDAYS']))
Features = xdata

# print("\n[Info] Translate into ndarray(%s) with shape=%s" % (ndarray.__class__, str(ndarray.shape)))  
# print("\n[Info] Show top 2 records:\n%s\n" % (ndarray[:2]))  
#print("\n[Info] ndarray:")
#print(ndarray)

# print("\n[Info] Label:")
# print(Label)
# print("\n[Info] Features:")
# print(Features)



 # 特徵欄位進行標準化 
#print("\n[Info] Normalized features...")  
from sklearn import preprocessing  
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1)) 
scaledFeatures = minmax_scale.fit_transform(Features)  
# scaler = preprocessing.StandardScaler()
# scaledFeatures= scaler.fit_transform(Features) 
# print("\n[Info] Show top 2 records:\n%s\n" % (scaledFeatures[:2])) 

train_features = scaledFeatures
train_labels =Label

# print("\n[Info] Features:")
# print(train_features)


#建立模型

import tensorflow as tf

# 1.建立model :使用Sequential model 
#將Layer放入Model中 (使用keras.layers.Dropout 防止過度擬合 ex:Dropout(0.2)隨機拋棄20%的數據(0-1之間) )
#input_shape:N (N維向量)
model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=8,input_shape=[train_features.shape[1]]),   
        tf.keras.layers.Dense(units=32),  
        tf.keras.layers.Dense(units=64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=1)
        ])




#2.以complie函數定義 損失函數(loss),優化函數(optimizer) 及成效衡量指標(metrics)
# 損失函數(loss) : mse(mean_squared_error) ,categorical_crossentropy,binary_crossentropy,mean_absolute_error
#kernel_initializer:權重的初始值(初始值的選擇會影響優化結果)
#verbose=2:每個epoch輸出一行紀錄
#epochs :訓練輪數
#batch_size:每個batch包含的樣本數
# model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.01), metrics=['accuracy'] ) 

# #3.執行訓練,訓練過程儲存在 train_history 
# train_history = model.fit(x=train_features, y=train_labels, validation_split=0.2, epochs=50, batch_size=30, verbose=2)
# print("Finished training the model")
# import loss_plot
# import acc_plot 
# loss_plot.draw(train_history)
# acc_plot.draw(train_history) 

# 2.以complie函數定義
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.01) )   
#3.執行訓練,訓練過程儲存在 train_history 
train_history = model.fit(train_features, train_labels, epochs=25, verbose=True)
import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(train_history.history['loss'])
plt.show()



# # 建立模型
# print("\n[Info] 建立模型")  
# from tensorflow.keras.models import Sequential #引入Sequential函式
# from tensorflow.keras.layers import Dense,Dropout #引入層數及
  
# model = Sequential()  
# # 輸入層
# #model.add(Dense(units=3, input_dim=1, kernel_initializer='uniform', activation='relu'))
# #kimbal調整如下:
# #model.add(Dense(units=3, input_shape=(train_features.shape[1],), kernel_initializer='uniform', activation='relu'))
# model.add(Dense(units=3, input_shape=(train_features.shape[1],), kernel_initializer='uniform', activation='relu'))
# #model.add(Dense(units=10, input_dim=9, kernel_initializer='random_uniform', activation='relu'))

# # 隱藏層
# #model.add(Dense(units=5, kernel_initializer='uniform', activation='relu'))
# model.add(Dense(units=20, kernel_initializer='random_uniform', activation='relu'))
# model.add(Dense(units=30, kernel_initializer='random_uniform', activation='relu'))
# model.add(Dense(units=40, kernel_initializer='random_uniform', activation='relu'))


# # 輸出層
# model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
# #print("\n[Info] Show model summary...")  
# #model.summary()



# # 進行訓練
# #https://keras.io/zh/
# #https://keras.io/zh/getting-started/sequential-model-guide/#keras-sequential
# print("\n[Info] 訓練中...")  


# # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
# import tensorflow.keras as ks
# model.compile(loss='binary_crossentropy', optimizer=ks.optimizers.Adam(0.001), metrics=['accuracy'] )   

# #train_history = model.fit(x=train_features, y=train_labels, validation_split=0.1, epochs=10, batch_size=30, verbose=2)  
# train_history = model.fit(x=train_features, y=train_labels, validation_split=0.1, epochs=50, batch_size=30, verbose=2)  

# #val_df = pd.read_csv(r'D:\Project\MyPython\titanic\data\val.csv')  
# #val_features, val_labels = data_preparer.preprocess(val_df)
# #train_history = model.fit(x=train_features, y=train_labels, validation_data=(val_features, val_labels), epochs=50, batch_size=30, verbose=2)  
# #print("\n[Info] 訓練成效 (文字)")  
# #print(train_history.history)


# # 評估模型
# #loss_val, acc_val, mse_val = model.evaluate(val_features, val_labels)
# #print(f"\n評估模型 : Loss is {loss_val},\nAccuracy is {acc_val * 100},\nMSE is {mse_val}")

# # 顯示結果
# #print("\n[Info] 訓練成效 (圖表)")
# import loss_plot
# import acc_plot 
# loss_plot.draw(train_history)
# acc_plot.draw(train_history)  
