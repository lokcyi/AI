#隨機森林
#很多決策樹，彼此都互相獨立，最後用投票決定最終模型
#包含多顆決策樹的分類器

#離散型投票決定:取眾數
#連續型投票決定:取平均

#隨機森林的原理:Bagging
#boostrap方法抽樣，抽sample 再抽feature，建立出一棵決策樹

#優缺點
#S:連續/離散都可以用也可以處理高維度特徵，不容易overfitting
#T:

from sklearn.ensemble import RandomForestClassifier #引入分類器
from sklearn import datasets
from sklearn.model_selection import train_test_split #引入如何區分測試集與訓練集
import matplotlib.pyplot as plt



iris=datasets.load_iris()
X=iris.data #鳶尾花四個特徵資料
y=iris.target # 三種不同類型的鳶尾花

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3) #分好訓練集與測試集
rfc=RandomForestClassifier(n_estimators=5,random_state=0) #隨機森林分類器 選擇要用多少棵決策樹 =5，隨機種子要固定下來，比較能夠確定優化結果是否是真的增加準確度
rfc = RandomForestClassifier(n_estimators=5)

rfc.fit(X_train,y_train)
y_predict=rfc.predict(X_test)
rfc.score(X_test,y_test) #output 預測成功率 0.93

#====================================================
#====================================================
rfc=RandomForestClassifier(n_estimators=100) 
#隨機森林分類器 選擇要用多少棵決策樹 =100==> 決策數的樹木
rfc.fit(X_train,y_train)
y_predict=rfc.predict(X_test)
rfc.score(X_test,y_test) #output 預測成功率 0.95(成功率有提升)

#====================================================
#====================================================
rfc=RandomForestClassifier(n_estimators=100,n_jobs=-1,random_state=50,min_samples_leaf=10) 
#隨機森林分類器 選擇要用多少棵決策樹 =100==> 決策數的樹木
#n_jobs =-1 :多核心有多少就用多少核心一起運算
#min_sample_leaf: 修剪樹枝，生長完後最後做樹枝的修剪 最少包含10個資訊量

rfc.fit(X_train,y_train)
y_predict=rfc.predict(X_test)
rfc.score(X_test,y_test) #output 預測成功率 0.93(成功率有提升)

#====================================================
#Feature Importance 判斷特徵的重要程度
#====================================================
imp=rfc.feature_importances_ #output(0.088,0.019,0.411,0.479) 依據代表每個特徵的重要性
imp
names=iris.feature_names #output 特徵 sepal length,sepal width,petal length,petal width
zip(imp,names)
imp,names=zip(*sorted(zip(imp,names))) #先排序
plt.barh(range(len(names)),imp,align='center')
plt.yticks(range(len(names)),names)
plt.xlabel('Importance of Features')
plt.ylabel('Features')
plt.title('Importance of Each Feature')
plt.show()




print('OK')

