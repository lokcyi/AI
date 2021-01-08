#(九) 結合不同類型的弱學習器
#VotingClassifier
from sklearn.ensemble import VotingClassifier#連續型資料 要用VotingRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris=datasets.load_iris()
X=iris.data #鳶尾花四個特徵資料
y=iris.target # 三種不同類型的鳶尾花

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0) #拆分樣本為訓練集(70%)與測試集(30%)

# 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

model_list=[]
m1=SVC() #模型 可以設定餐數調整
model_list.append(('svm',m1))
m2=DecisionTreeClassifier()
model_list.append(('DT',m2))
m3=GaussianNB()
model_list.append(('NB',m3))
vc=VotingClassifier(model_list)

vc.fit(X_train,y_train)
vc.predict(X_test)
vc.score(X_test,y_test) # 準確程度 output (0.97)

plt.scatter(X[:,2],X[:,3],c=y)
plt.show()
plt.scatter(X[:,2],X[:,3],c=vc.predict(X)) #針對全部樣本作predict
plt.show()

print('OK')



