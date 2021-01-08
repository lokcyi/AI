#多元迴歸 特徵值 >1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model #線性回歸
from sklearn.preprocessing import PolynomialFeatures #多項式迴歸
from sklearn.pipeline import make_pipeline 
from sklearn.datasets import make_regression # 資料集
from sklearn.model_selection import train_test_split
 
X,y=make_regression(n_samples=100,n_features=5,noise=20) #100個樣本點 特徵=5 , 建造多特徵資料

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
print("score train :",regr.score(X_train,y_train)) 
print("score test:",regr.score(X_test,y_test))  
# #====================
 #除了房屋坪數大小的特徵外，我們再加入距離市區遠近的特徵
# #====================
size=[5,10,12,14,18,30,33,55,65,80,100,150]
distance=[50,20,70,100,200,150,30,50,70,35,40,20]
price=[300,400,450,800,1200,1400,2000,2500,2800,3000,3500,9000]
series_dict={'X1':size,'X2':distance,'y':price}
df=pd.DataFrame(series_dict)
X=df[['X1','X2']] # 增加X2
y=df[['y']]

regr=linear_model.LinearRegression()
regr.fit(X, y)
regr.score(X,y) #0.90

w_0=regr.intercept_ 
w_1=regr.coef_  
print("intercept :",w_0) # 截距309.68
print("cofficient:",w_1)  # cofficient  [51.943,1.622] 代表 Size 比 distance 重要


