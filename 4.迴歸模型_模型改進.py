 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model #線性回歸
from sklearn.preprocessing import PolynomialFeatures #多項式迴歸
from sklearn.pipeline import make_pipeline 
from sklearn.datasets import make_regression # 資料集
from sklearn.model_selection import train_test_split
 

#==============
# 模型改進
#     在成本函數  或損失函數 上加上額外的懲罰項
#     LASSO REGRESSION : 懲罰項就是係數絕對值得合越小越好，越不重要的係數會=0，同時也可以達到特徵篩選
#     RIDGE REGRESSION : 懲罰項 係數的平方相加，藉由係數變小，達到避免過度擬合
#==============
X,y=make_regression(n_samples=1000,n_features=10,noise=10) #download 資料 100個樣本點 特徵=5 , 建造多特徵資料 

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3) #把資料 區分好訓練集與測試集

#====================
#(四)多元迴歸 Multivariable Regression
#====================
regr=linear_model.LinearRegression()
regr.fit(X_train,y_train) #用 linear_model 裡面的LinearRegression對訓練集進行模型建構:

#如何叫出 那條線 線性迴歸 的 截距w0 與 斜率w1  y=w0+w1*X: 
w_0=regr.intercept_ #intercept 截距 2.72
w_1=regr.coef_  #cofficient : [27.78890155 79.46213354 59.93566859 40.85498441 88.85502858]
print("intercept :",w_0) 
print("cofficient:",w_1)  
regr.score(X_train,y_train)
regr.score(X_test,y_test)
print("Traning score  :",regr.score(X_train,y_train)) 
print("Testing score  :",regr.score(X_test,y_test))  


# 兩種避免過度擬和的模型，分別為Lasso Regression與Ridge Regression
print('---------------------')
# 1. Lasso Regression
clf_lasso=linear_model.Lasso(alpha=0.5) #參數越大懲罰項越高 預設1， 越大fit程度越低
clf_lasso.fit(X_train,y_train)
print("clf_lasso Traning score  :",clf_lasso.score(X_train,y_train)) 
print("clf_lasso Testing score  :",clf_lasso.score(X_test,y_test))  
print('---------------------')
# 2. Ridge Regression
clf_ridge=linear_model.Ridge(alpha=0.5)
clf_ridge.fit(X_test,y_test)
print("clf_ridge Traning score  :",clf_ridge.score(X_train,y_train)) 
print("clf_ridge Testing score  :",clf_ridge.score(X_test,y_test))  

#================================
# 多項式模型
#================================
size=[5,10,12,14,18,30,33,55,65,80,100,150]
price=[300,400,450,800,1200,1400,2000,2500,2800,3000,3500,9000]
plt.scatter(size,price)
plt.show()

#=============
#開始建模。先全部假設都為訓練集
#================
series_dict={'X':size,'y':price}
df=pd.DataFrame(series_dict)
X=df[['X']]
y=df[['y']]

model=make_pipeline(PolynomialFeatures(4),linear_model.Ridge()) #在PolynomialFeatures裡面放入需要的維度，比方說我們放入3，指的就是三次方的多項式
model.fit(X,y) #建構模型
plt.scatter(X,y)
plt.plot(X,model.predict(X),color='red')
plt.show()

model=make_pipeline(PolynomialFeatures(4),linear_model.Lasso()) #在PolynomialFeatures裡面放入需要的維度，比方說我們放入3，指的就是三次方的多項式
model.fit(X,y) #建構模型
plt.scatter(X,y)
plt.plot(X,model.predict(X),color='red')
plt.show()
