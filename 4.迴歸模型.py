#回歸模型  
# 1.線性迴歸 LINEAR REGRESSION
# 2.多項式線性迴歸 PLOYNOMIAL REGRESSION
# 3.多變量迴歸 

# 最小平方法 極小化誤差
# y(x) =w0+w1

# 梯度下降法 一路找坡度最陡的地方 =>batch gradient descent

# 當樣本集太大 ==> stochastic gradient descent (隨機，計算快)

# 低度擬合&過度擬合
#     特徵數量不太少 或過多(ex特徵比樣本還要多) 可能找成過度擬合 

# 模型改進
#     在成本函數  或損失函數 上加上額外的懲罰項
#     LASSO REGRESSION : 懲罰項就是係數絕對值得合越小越好，越不重要的係數會=0，同時也可以達到特徵篩選
#     RIDGE REGRESSION : 懲罰項 係數的平方相加，藉由係數變小，達到避免過度擬合

#===============
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model #線性回歸
from sklearn.preprocessing import PolynomialFeatures #多項式迴歸
from sklearn.pipeline import make_pipeline 
from sklearn.datasets import make_regression # 資料集
from sklearn.model_selection import train_test_split
 

#X,y=make_regression(n_samples=100,n_features=1,noise=0) #100個樣本點 特徵=1 ,

X,y=make_regression(n_samples=100,n_features=1,noise=20) #100個樣本點 特徵=1 ,
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3) #把資料 區分好訓練集與測試集


#====================
#Simple Linear Regression 簡單線性回歸
#====================
regr=linear_model.LinearRegression()
regr.fit(X_train,y_train) #用 linear_model 裡面的LinearRegression對訓練集進行模型建構:


#如何叫出 那條線 線性迴歸 的 截距w0 與 斜率w1  y=w0+w1*X: 
w_0=regr.intercept_ #intercept 截距 3.77
w_1=regr.coef_  #cofficient 標準化迴歸係數(Standardized regression cofficient)
print("intercept :",w_0) #-3.7
print("cofficient:",w_1) #66.7

#====================
#show chart 模型已經建構完成了! 接下來，我們來看看訓練集的成果。
#====================
plt.scatter(X_train,y_train,color='black') #scatter散佈圖
plt.scatter(X_test,y_test,color='red')
plt.title('Train and Test')
plt.show()

plt.scatter(X_test,y_test,color='red') 
#plt.scatter(X_test,regr.predict(X_test),color='blue')   #模型畫成 scatter散佈圖
plt.plot(X_test,regr.predict(X_test),color='blue',linewidth=1) #模型畫成 線性模型 plot()畫出曲線  藍色的線就是我們建構出的線性迴歸
plt.title('TEST Model Result y=w0+w1*X ')
plt.show()


#====================
#透過score來看模型成效  如果訓練集的分數很高，但測試集的分數卻很低，那就是…過度擬和啦!
#====================
regr.score(X_train,y_train) #0.86 
 
regr.score(X_test,y_test) #0.79  

#===============================================
#Gradient Decent
#===============================================
alpha=0.001
repeats=1000
w0=0
w1=0
errors=[]
points=[]

for j in range(repeats):
    error_sum=0
    squared_error_sum=0
    error_sum_x=0
    for i in range(len(X_train)):
        predict=w0+(X_train[i]*w1)
        squared_error_sum=squared_error_sum+(y_train[i]-predict)**2
        error_sum=error_sum+y_train[i]-predict
        error_sum_x=error_sum_x+(y_train[i]-predict)*X_train[i]
    w0=w0+(alpha*error_sum)
    w1=w1+(alpha*error_sum_x)
    errors.append(squared_error_sum/len(X_train))

print('w0: %2f' %w0) #-3.77
print('w1: %2f' %w1) #41.89
#===============================================
predicts=[]
mean_error=0
for i in range(len(X_test)):
    predict=w0+(X_test[i]*w1)
    predicts.append(predict)

plt.scatter(X_test,predicts)
plt.scatter(X_test,y_test,color='red')
plt.title('TEST')
plt.show()

print('OK')

#======================================
# 多項式迴歸 Polynomial Regression 
#======================================

size=[5,10,12,14,18,30,33,55,65,80,100,150]
price=[300,400,450,800,1200,1400,2000,2500,2800,3000,3500,9000]
plt.scatter(size,price)
plt.show()