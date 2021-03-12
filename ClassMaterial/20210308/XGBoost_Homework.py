# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np 
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import plot_importance 

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
# df_train_eh =pd.get_dummies(df_train.drop(['MOVE_QTY','MFG_DATE','AI'], axis=1),columns= [ 'TOOLG_ID'])
X_dropped = np.asarray(df_train.drop(['TOOLG_ID','MOVE_QTY','MFG_DATE','AI'], axis=1))
Y_dropped = np.asarray(df_train['MOVE_QTY'])

X_train, X_val, y_train, y_val = train_test_split(X_dropped, Y_dropped, random_state=1) #Seed： 亂數種子，可以固定我們切割資料的結果random_state=777,
#調參------------------------------------------------------------
cv_params = {'n_estimators': [400, 500, 600, 700],'max_depth':[3,5,7],'min_child_weight':[1,3,5]}
other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
model = xgb.XGBRegressor(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)

optimized_GBM.fit(X_train, y_train) 

test_score = optimized_GBM.score(X_val,y_val)

print('test 得分:{0}'.format(test_score))
# evalute_result = optimized_GBM.grid_scores_
# print('每輪迭代執行結果:{0}'.format(evalute_result))
 
print('引數的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
print('cv_results_',optimized_GBM.cv_results_)


#===================================================
# 方法1
#===================================================
# # # 找出最重要的特徵 (值越大，特徵越重要)
 
# other_params = {'learning_rate': 0.1, 'n_estimators': 400, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
# 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
# model = xgb.XGBRegressor(**other_params)
# model.set_params(n_estimators=700) 
# model.set_params(min_child_weight=5) 
# model.set_params(max_depth=9) 
#引數的最佳取值：{'max_depth': 9, 'min_child_weight': 5, 'n_estimators': 700}
#===================================================
# 方法2
#===================================================
bst_model = optimized_GBM.estimator
print(bst_model)

bst_model.fit(X_train, y_train)
plot_importance(bst_model)#,max_num_features=10)
plt.show()

 


# %%
xgb.to_graphviz(bst_model,num_trees=20)  #可以透過 num_trees 的設定去看不同顆樹的分裂結果，以下是第 1 棵樹 ( num_trees = 0 )的結果：


# %%
xgb.plot_tree(bst_model, num_trees=20)
fig.savefig('xgb_tree.jpg')


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
print(df.head())        


# %%
model


# %%
X_droppedtest = np.asarray(df)
# dtest = xgb.DMatrix(X_droppedtest)
# y_predict = model.predict(X_droppedtest)
y_predict = model.predict(X_droppedtest)
df['predict'] = y_predict # 預測WIP 數
plt.plot( df['NO_HOLD_QTY'] ,y_predict )
plt.show()


# %%



