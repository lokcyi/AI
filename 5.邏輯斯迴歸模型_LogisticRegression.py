#邏輯斯迴歸模型 Logistic Regression
#邏輯斯迴歸是 ==> 處理二元的問題 (不是0就是1的問題 漲/跌)，屬於線性分類器的一種

#Odds 勝算比(事情發生的機率/事情沒發生的機率  Odds = p/(1-p))
#logit(Odds) = ln(p/1-p) 

#logit(Odds) 數字要經過==> Sigmoid函數 的轉換，讓數字介於0~1 且中間的遞增式很快的 就很適合做轉換

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

data=pd.read_csv('LogR_data.csv')
X=data['Amount'].values # 每個人每天喝n杯飲料
y=data['Result'].values
X.shape
X=X.reshape(-1,1)  # 改成矩陣 1列
X.shape

model=linear_model.LogisticRegression()
model.fit(X,y)
print('coef',model.coef_)
print('intercept',model.intercept_)
w1=float(model.coef_) #coef 係數
w0=float(model.intercept_) #intercept 級距

#做sigmoid轉換 讓數字介於0~1 且中間的遞增式很快的 就很適合做轉換
def sigmoid(x,wo,w1):
    ln_odds=wo+w1*x
    return 1/(1+np.exp(-ln_odds))
#========================

x=np.arange(0,10,1)
s_x=sigmoid(x,w0,w1)
plt.plot(x,s_x)
plt.axhline(y=0.5, ls='dotted', color='k')
plt.show()
#=========================================================
print('result',model.predict([[0],[1],[2],[3]])) #預測每天喝 0 1 2 3 是否會得到糖尿病 ==>result [0 0 1 1]
model.predict_proba(X) #可以看分析出來 得到的機率 ，因而決定是否會得到。
# array([[0.68862321, 0.31137679], #得到糖尿病的機率 ,不會得到糖尿病的機率 ==> 何者大 決定不會不得糖尿病
#        [0.54212644, 0.45787356],
#        [0.25338066, 0.74661934],
#        [0.38796409, 0.61203591],
#        [0.54212644, 0.45787356],
#        [0.54212644, 0.45787356],
#        [0.68862321, 0.31137679],
#        [0.68862321, 0.31137679],
#        [0.54212644, 0.45787356],
#        [0.68862321, 0.31137679],
#        [0.38796409, 0.61203591],
#        [0.25338066, 0.74661934],
#        [0.25338066, 0.74661934],
#        [0.38796409, 0.61203591],
#        [0.38796409, 0.61203591],
#        [0.68862321, 0.31137679],
#        [0.38796409, 0.61203591],
#        [0.25338066, 0.74661934],
#        [0.38796409, 0.61203591],
#        [0.25338066, 0.74661934]])

model.score(X,y) #成功分類 0.75

#===

