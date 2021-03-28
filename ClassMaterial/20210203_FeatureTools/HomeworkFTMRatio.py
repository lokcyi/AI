# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#多項式迴歸分析 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model #線性回歸
from sklearn.preprocessing import PolynomialFeatures #多項式迴歸
from sklearn.pipeline import make_pipeline 
from sklearn.datasets import make_regression # 資料集
from sklearn.model_selection import train_test_split
 

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
model=make_pipeline(PolynomialFeatures(3),linear_model.LinearRegression()) #在PolynomialFeatures裡面放入需要的維度，比方說我們放入3，指的就是三次方的多項式
model.fit(X,y) #建構模型

plt.scatter(X,y)
plt.plot(X,model.predict(X),color='red')
plt.show()

#=============
#在多項式迴歸外加上個迴圈，這樣就可以很清楚的看到不同次方迴歸模型的表現
#=============
scores=[]
colors=['green','purple','gold','blue','black']
plt.scatter(X,y,c='red')
for count,degree in enumerate([1,2,3,4,5]):
    model=make_pipeline(PolynomialFeatures(degree),linear_model.LinearRegression())
    model.fit(X,y)
    scores.append(model.score(X,y))
    plt.plot(X,model.predict(X),color=colors[count],label='degree %d' %degree)

plt.legend(loc=2)
plt.show()

print('scores ', scores)


