# 2.決策數模型 inductive learning ，是歸納學習 

# 決策數運作原理
#     衡量特徵減少Entropy（熵）(不確定性:0=穩定,1=不穩定)的程度，決策樹根到葉 要越來越穩定
#     資訊增益(information Gain): 資訊增益越多越好。


# Training Data ，Test Data，trimDATA

from sklearn import tree
from sklearn import datasets
import pydotplus #用來看樹怎麼做分類 conda install -c conda-forge pydotplus
iris=datasets.load_iris()
X=iris.data #鳶尾花四個特徵資料
y=iris.target # 三種不同類型的鳶尾花

#====================================================
#決策數模型輸出
#====================================================
clf=tree.DecisionTreeClassifier(criterion='entropy').fit(X,y)
clf.score(X,y)  #output 1.0

dot_data=tree.export_graphviz(clf,out_file=None)
graph=pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('iris.pdf')
#====================================================
#拆分訓練集與測試集(避免overfitting)
#====================================================
from sklearn.model_selection import train_test_split #自動分類 訓練集與測試集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

clf=tree.DecisionTreeClassifier(criterion='entropy').fit(X_train,y_train)
clf.score(X_train,y_train)  #output 1.0

clf.predict(X_test) #entropy模型
clf.score(X_test,y_test) #output(0.9)

#====================================================
#過度配適初步調整
#====================================================
clf=tree.DecisionTreeClassifier(criterion='entropy',max_depth=3).fit(X_train,y_train) #max_depth = 3 讓決策樹最多只能長三層
clf.predict(X_test) #entropy模型
clf.score(X_train,y_train) #訓練集 0.99
clf.score(X_test,y_test) #測試集 0.93

#====================================================
#如果節點採用Gini
#====================================================
clf=tree.DecisionTreeClassifier(criterion='gini',max_depth=3).fit(X_train,y_train) #max_depth = 3 讓決策樹最多只能長三層
clf.predict(X_test) #gini模型
clf.score(X_train,y_train) #訓練集 0.93
clf.score(X_test,y_test) #測試集 0.93
dot_data=tree.export_graphviz(clf,out_file=None)
graph=pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('iris_gini.pdf')

print('OK')
    