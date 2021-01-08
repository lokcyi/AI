#(七) Ensemble Learning(學習集成) 主要是透過結合多個機器學習器而成的大模型
#集成式學習:透過多個機器學習器的結果 透過不同的方法宗合起來得到最終的結果
#方法包含三種:Bagging,Bosting及Stacking
#1.Bagging法 :隨機抽取的概念。已抽取的內容建構機器學習模型。再將抽取的樣本放回，再抽一次 第二次建模(採用不同的特徵)
#2.這個取後放回的抽取方式 稱之為 bootstrap

#最後透過每個建模產出的結果，投票決定最終模型，一人一票等權重加權分式，
#如果目標為連續型，則改採對每個學習器的結果取平均
#隨機抽樣本 隨機抽特徵
#離散型資料 : 取眾數 作為結果
#連續型資料 : 取平均 作為結果

#1.抽幾次模型&每次數量多少
#2.抽幾個特徵

from sklearn.ensemble import BaggingClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Juypter=> %matplotlib inline , vsCode =>plt.show()

iris=datasets.load_iris()
X=iris.data #鳶尾花四個特徵資料
y=iris.target # 三種不同類型的鳶尾花

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
len(X)#output 150 個樣本
len(X_train)#output 105 個樣本(取70%做訓練集)
len(X_test)#output 45 個樣本(取30%做測試集)


#子模型=>決策數 
from sklearn import tree 

clf=tree.DecisionTreeClassifier() #用預設參數
bagging=BaggingClassifier(base_estimator=clf,n_estimators=10,
                    bootstrap=True,bootstrap_features=True,max_features=3,max_samples=0.7) 
                    #1.子模型 2.產生多少個子模型 (10個決策樹) 3.是否要取後放回(true)  4. 抽出多少特徵(百分比或整數) 5.一次要抽出多少樣本(70%) 

bagging.fit(X_train,y_train) #
bagging.predict(X_test) #FIT後就可以predict ，可以對測試集做測試 預測出來測試集樣本 對應到哪些類別的鳶尾花
bagging.score(X_train,y_train)#訓練集的資料 準確程度(output 0.98)
bagging.score(X_test,y_test)#訓練集的資料 準確程度(output 0.977)

plt.scatter(X[:,2],X[:,3],c=y)
plt.show()

plt.scatter(X[:,2],X[:,3],c=bagging.predict(X))
plt.show()

 

#================================================
from sklearn import svm
clf=svm.LinearSVC() #用預設參數
bagging=BaggingClassifier(base_estimator=clf,n_estimators=10,
                    bootstrap=True,bootstrap_features=True,max_features=3,max_samples=0.7) 
                    #1.子模型 2.產生多少個子模型 (10個決策樹) 3.是否要取後放回(true)  4. 抽出多少特徵(百分比或整數) 5.一次要抽出多少樣本(70%) 

bagging.fit(X_train,y_train) #
bagging.predict(X_test) #FIT後就可以predict ，可以對測試集做測試 預測出來測試集樣本 對應到哪些類別的鳶尾花
bagging.score(X_train,y_train)#訓練集的資料 準確程度(output 0.98)
bagging.score(X_test,y_test)#訓練集的資料 準確程度(output 0.977)

plt.scatter(X[:,2],X[:,3],c=y) #用第三 與 第四個特徵
plt.show()
plt.scatter(X[:,2],X[:,3],c=bagging.predict(X))
plt.show()
 


#================================================
from sklearn.naive_bayes import GaussianNB #高斯貝式分類器
clf=GaussianNB()  
bagging=BaggingClassifier(base_estimator=clf,n_estimators=10,
                    bootstrap=True,bootstrap_features=True,max_features=3,max_samples=0.7) 
                    #1.子模型 2.產生多少個子模型 (10個決策樹) 3.是否要取後放回(true)  4. 抽出多少特徵(百分比或整數) 5.一次要抽出多少樣本(70%) 

bagging.fit(X_train,y_train) #
bagging.predict(X_test) #FIT後就可以predict ，可以對測試集做測試 預測出來測試集樣本 對應到哪些類別的鳶尾花
bagging.score(X_train,y_train)#訓練集的資料 準確程度(output 0.98)
bagging.score(X_test,y_test)#訓練集的資料 準確程度(output 0.977)

plt.scatter(X[:,2],X[:,3],c=y)
plt.show()
plt.scatter(X[:,2],X[:,3],c=bagging.predict(X))
plt.show()
print('OK')

