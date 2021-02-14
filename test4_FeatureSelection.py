#(四) Feature Selection 特徵選擇 =>特徵過多
#特徵越多越好，容易得到好的預測效果，但過多的特徵，除了運算成本增加，更容易造成過度擬和、維度災難等問題

from sklearn import feature_selection as fs
from sklearn import datasets
#===============================================================
#方法 1. 設定變異數門檻值，剔除變異過低的特徵

X=[[0,0,1],[0,1,0],[1,0,0],[0,1,1],[0,1,0],[0,1,1]]
#設定變異數門檻值 柏努力分配 變異數(1*(1-p))
#Bernoulli distribution 變異數 Var(X)=p(1-P)
sel=fs.VarianceThreshold(threshold=0.8*(1-0.8)) 
result = sel.fit_transform(X)
#result 變成二維陣列
#===============================================================
#方法 2 Univariate feature selection
# 透過針對每個特徵 與 Target之間的 做個別的計算  的統計值來決定
# 最重要K個的特徵(selectKBest) / 選取排名前 K% 的重要特徵(selecPercentile)
# For regression
# [target 是連率值] => f_regression,mutual_info_regression
# [分類的問題]  => chi2,f_classif,mutual_info_classif

#選擇要用的統計檢定方式，與要多少個有用的特徵

iris=datasets.load_iris()
X=iris.data #鳶尾花四個特徵資料
y=iris.target # 三種不同類型的鳶尾花

X.shape #output(150,4) 150個樣本 4個特徵

#作法1 4=>3
X_new=fs.SelectKBest(fs.chi2,k=3).fit_transform(X,y) # 卡方分配 + 篩選三個重要特徵
X_new.shape #output(150,3) 150個樣本 3個特徵

#作法2 50%
#mutual infomation :用來衡量兩個數據分布的吻合程度 
X_new=fs.SelectPercentile(fs.mutual_info_classif,percentile=50).fit_transform(X,y) # mutual information(互信息)  前50%個重要特徵
X_new.shape #output(150,2) 150個樣本 2個特徵

print('OK')