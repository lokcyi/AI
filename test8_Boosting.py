#(八) Ensemble Learning(學習集成) 針對錯誤不斷訓練的boosting
#跟bagging一樣  boosting同樣會集結多個分類器，不同的是boosting方法裡的分子分類器各自獨立
#bootsting裡的分類器則會由前一個分類器的結果而做更進一步的修正，
#Bagging 最後的結果採一人一票等值的方式進行，Boosting 裡面則是菁英體制，準確度高的給予更高的權重。

#初始每個樣本抽到機率 一樣(1/M)，進行T輪的迭代學習，
#每一次迭代都會根據現在的樣本抽到的機率使用弱學習器進行學習，得到這次迭代的模型。
#然後再去計算迭代模型的誤差，更新每個樣本被抽到的機率，在進行下一輪。

#增加不容易預測樣本的機率，目的是為了把預測不準的樣本變成準確
#模型越準 權重會越高
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris=datasets.load_iris()
X=iris.data #鳶尾花四個特徵資料
y=iris.target # 三種不同類型的鳶尾花

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0) #拆分樣本為訓練集(70%)與測試集(30%)

#AdaBoostClassifier  (learning_rate:子模型權重縮減系數)
#AdaBoostRegressor

#GradientBoostingClassifier
#GradientBoostingRegressor
#===================================================
adb = AdaBoostClassifier() # base_estimator = 用decision Tree
adb.fit(X_train,y_train)
adb.predict(X_test) #做預測結果
adb.score(X_test,y_test) #output(0.91)
plt.scatter(X[:,2],X[:,3],c=y)
plt.show()
plt.scatter(X[:,2],X[:,3],c=adb.predict(X))
plt.show()
#===================================================
from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
adb=AdaBoostClassifier(base_estimator=clf,learning_rate=0.5,n_estimators=100)
adb.fit(X_train,y_train)
adb.predict(X_test) #做預測結果
adb.score(X_test,y_test) #output(0.91)
plt.scatter(X[:,2],X[:,3],c=y)
plt.show()
plt.scatter(X[:,2],X[:,3],c=adb.predict(X))
plt.show()
#===================================================
#GradientBoost
#===================================================
gb=GradientBoostingClassifier() #loss=deviance 損失函數
gb.fit(X_train,y_train)
gb.predict(X_test)
gb.score(X_test,y_test) #準確程度
plt.scatter(X[:,2],X[:,3],c=y)
plt.show()
plt.scatter(X[:,2],X[:,3],c=gb.predict(X))
plt.show()
print('OK')

