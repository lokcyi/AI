#%%
import tensorflow as tf
import numpy as np 
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

save_model = 'D:/ML/training_model.h5'

def Training(source_file_path):
    df = pd.read_excel(source_file_path)
    #df = pd.read_csv(source_file_path)

    df['PRIORITY'] = df['PRIORITY'].astype(str)
    df['IS_MAIN_ROUTE'] = df['IS_MAIN_ROUTE'].astype(str)

    df['DATA_DATE'] = df['DATA_DATE'].astype(str)
    df = df.loc[df['LOT_TYPE']=='FDY']
    df = df.loc[df['LAYER']!='XX']

    # df.fillna(0, inplace=True)
    df = df.assign(PROCESSED_DAYS = ((pd.to_datetime(df['DATA_DATE'], format='%Y%m%d')-pd.to_datetime(df['WS_DATE'], format='%Y%m%d'))/pd.Timedelta(1, 'D')).fillna(0).astype(int))
    df = df.assign(REMAIN_DAYS = ((pd.to_datetime(df['ACTUAL_WP_OUT'], format='%Y%m%d')-pd.to_datetime(df['DATA_DATE'], format='%Y%m%d'))/pd.Timedelta(1, 'D')).fillna(0).astype(int))

    df2 = df.drop(columns=['IDX','LOT_TYPE','WS_DATE','ACTUAL_WP_OUT','DATA_DATE','LAYER','LOT_ID'])

    labelencoder = LabelEncoder()
    df2['PRIORITY'] = labelencoder.fit_transform(df2['PRIORITY'])
    df2['IS_MAIN_ROUTE'] = labelencoder.fit_transform(df2['IS_MAIN_ROUTE'])
    df2['STATUS'] = labelencoder.fit_transform(df2['STATUS'])
    df2['CHIPNAME'] = labelencoder.fit_transform(df2['CHIPNAME'])
    df2['OP_NO'] = labelencoder.fit_transform(df2['OP_NO'])

    df3 = df2.drop(df2.loc[:, '0I':'UG'].columns, axis = 1) 
    # df3.info()

    X_dropped = np.asarray(df3.drop('REMAIN_DAYS', axis=1))
    Y_dropped = np.asarray(df3['REMAIN_DAYS'])

    #將Layer放入Model中
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=8,input_shape=[X_dropped.shape[1]]),
        tf.keras.layers.Dense(units=32),
        tf.keras.layers.Dense(units=64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=1)
        ])

    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.01))

    history = model.fit(X_dropped, Y_dropped, epochs=20, verbose=True)
    print("Finished training the model")
    model.save(save_model)

    plt.xlabel('Epoch Number')
    plt.ylabel("Loss Magnitude")
    plt.plot(history.history['loss'])
    plt.show()

def Testing(source_file_path):
    #df_test = pd.read_excel(source_file_path)
    df_test = pd.read_csv(source_file_path)
 
    df_test['PRIORITY'] = df_test['PRIORITY'].astype(str)
    df_test['IS_MAIN_ROUTE'] = df_test['IS_MAIN_ROUTE'].astype(str)

    df_test['DATA_DATE'] = df_test['DATA_DATE'].astype(str)
    df_test = df_test.loc[df_test['LOT_TYPE']=='FDY']
    df_test = df_test.loc[df_test['LAYER']!='XX']

    df_test = df_test.assign(PROCESSED_DAYS = ((pd.to_datetime(df_test['DATA_DATE'], format='%Y%m%d')-pd.to_datetime(df_test['WS_DATE'], format='%Y%m%d'))/pd.Timedelta(1, 'D')).fillna(0).astype(int))
    df_test = df_test.assign(REMAIN_DAYS = ((pd.to_datetime(df_test['ACTUAL_WP_OUT'], format='%Y%m%d')-pd.to_datetime(df_test['DATA_DATE'], format='%Y%m%d'))/pd.Timedelta(1, 'D')).fillna(0).astype(int))

    df_test2 = df_test.drop(columns=['IDX','LOT_TYPE','WS_DATE','ACTUAL_WP_OUT','DATA_DATE','LAYER','LOT_ID'])

    labelencoder = LabelEncoder()
    df_test2['PRIORITY'] = labelencoder.fit_transform(df_test2['PRIORITY'])
    df_test2['IS_MAIN_ROUTE'] = labelencoder.fit_transform(df_test2['IS_MAIN_ROUTE'])
    df_test2['STATUS'] = labelencoder.fit_transform(df_test2['STATUS'])
    df_test2['CHIPNAME'] = labelencoder.fit_transform(df_test2['CHIPNAME'])
    df_test2['OP_NO'] = labelencoder.fit_transform(df_test2['OP_NO'])

    df_test3 = df_test2.drop(df_test2.loc[:, '0I':'UG'].columns, axis = 1) 
    
    X_Test = np.asarray(df_test3.drop('REMAIN_DAYS', axis=1))
    Y_Test = np.asarray(df_test3['REMAIN_DAYS'])

    # df_test3.info()
    model = tf.keras.models.load_model(save_model)
    # print(model.predict(df_test3))

    # plt.xlabel('Seq Number')
    plt.ylabel("Remain Days")

    plt.plot(model.predict(X_Test), label = "Predict", color='red', marker='.',linewidth = '0.5')
    plt.legend()
    plt.show()
Training('D:/projects/ai/poc/homework/Training_Data.xlsx')
Testing('D:/projects/ai/poc/homework/Test_Data.xlsx')
# Training('D:/projects/ai/poc/homework/Training_Data.csv')
# Testing('D:/projects/ai/poc/homework/Test_Data.csv')
#Training('D:/projects/ai/poc/homework/Training_Data.xlsx')
# Testing('D:/Documents/機器學習/Data/Testing_Data.xlsx')

# %%
