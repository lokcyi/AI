
import pandas as pd
import numpy as np 
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
def readDataFromFile(file_path):
    df = pd.read_csv(file_path)
    return df

df_train=readDataFromFile('../../homework/training_data_20210302.csv')
df_train = df_train.loc[df_train['TOOLG_ID']=='PK_DUVKrF']
# 觀察缺失值------------------------------------------
print(df_train.isnull().sum())
# 做数据切分------------------------------------------
print(df_train.drop(['TOOLG_ID','MOVE_QTY','MFG_DATE','AI'], axis=1).describe())
df_train_eh =df_train.drop(['TOOLG_ID','MOVE_QTY','MFG_DATE','AI'], axis=1) 
X_dropped = np.asarray(df_train_eh)
Y_dropped = np.asarray(df_train['MOVE_QTY'])

X_train, X_val, y_train, y_val = train_test_split(X_dropped, Y_dropped, random_state=1)
xgTrain = xgb.DMatrix(X_train,y_train)
xgVal = xgb.DMatrix(X_val,y_val)

#調超參 best_nround-----------------------------------------------------------
 
other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
model = xgb.XGBRegressor(**other_params)
num_round = 10
best_nround = 1000

# bst = xgb.train(param, dtrain, num_round)
# 设定watchlist用于查看模型状态
watchlist  =[(xgTrain, 'train'), (xgVal, 'valid')]

res = xgb.cv(other_params,xgTrain, nfold=3,num_boost_round=500,metrics='rmse',early_stopping_rounds=25)
# #找到最佳迭代轮数
best_nround = res.shape[0] - 1
print('找到最佳迭代轮数',best_nround)
bst = xgb.train(other_params, xgTrain, best_nround, watchlist)


# %%
cols=['M','U','PT','UP_TIME','EQP_UTIL','TC','CS','C_AI']
df_testing = df_train.drop(['TOOLG_ID','MOVE_QTY','MFG_DATE','AI'], axis=1)
df_testing[cols]= df_testing[cols].mean()
df_testing =df_testing[0:1]
# max_wip = df_testing['NO_HOLD_QTY'].max()* .75
max_wip = np.percentile(df_testing['NO_HOLD_QTY'], 50) # return 50th percentile, e.g median.
tick = (df_testing['NO_HOLD_QTY'].max() - df_testing['NO_HOLD_QTY'].min()) /400
if max_wip <=0 :
    max_wip=100
if tick <10 :
    tick = 10
print(max_wip,tick)

df = pd.DataFrame(columns = df_testing.columns)

for i in range(1000):
    df_testing['NO_HOLD_QTY']=max_wip+ tick*i
    df = df.append(df_testing,ignore_index=True)
print(df.shape,df.head())        


# %%
X_droppedtest = np.asarray(df)
xgtest = xgb.DMatrix(X_droppedtest)
y_predict = bst.predict(xgtest)
df['predict'] = y_predict

plt.plot(df['NO_HOLD_QTY'] , y_predict)
plt.show()


# %%



