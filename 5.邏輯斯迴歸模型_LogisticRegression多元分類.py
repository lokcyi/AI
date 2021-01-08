#多元分類
#邏輯斯迴歸模型 主要處理 二元分類的問題，但如果有三個種類要怎麼進行
#1.分類器多做幾次 每次都拆成兩兩 [1,2+3] [2,1+3] [3,1+2]

from sklearn import datasets
from sklearn.cross_validation import train_test_split
iris=datasets.load_iris()
X=iris.data   #特徵
y=iris.target #目標
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3) #資料集分類
model=linear_model.LogisticRegression() #model =>邏輯斯迴歸模型 Logistic Regression
model.fit(X_train,y_train) 
model.predict(X_test) #訓練集測試出來的結果 
model.predict_proba(X_test) 
model.score(X_train,y_train) #訓練集的表現 0.93
model.score(X_test,y_test) #測試集的表現 0.93

