#(六) LDA (Linear Discriminant Analysis)線性判斷式分析
# LDA與PCA 都是透過映射將特徵由高維度轉換到低維度
#不同的是LDA同時考量了標籤(target)，屬於監督式學習
#PCA 保留了特徵最大的變異性
#LDA則是為了使降維後的資訊點更容易被區分

#PCA 屬於 無監督式學習，目的在保留特徵的變異性
#LDA 屬於 監督式學習  ，目的是讓降維之後的數據有更好的區分(類別跟類別之間距離越開越好，類別組內之間的關係越近越好)

#LDA的優勢是在降維的過程中可以使用類別的先驗知識，(一般情況 監督式學習會優於非監督式學習)_
#LAD也有一些 限制，比方說最多只能降維到類別數-1的維度(PCA沒有這個限制)。
#不論 LDA PCA 都是架設特徵維常態分配，對於非常態分配的特徵來說降維效果不一定這麼好

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import datasets
import pandas as import pd
import numpy as np

iris=datasets.load_iris()
X=iris.data #鳶尾花四個特徵資料
y=iris.target # 三種不同類型的鳶尾花

LDA=LinearDiscriminantAnalysis(n_components=2) #降維到2個特徵
lda_X_=LDA.fit_transform(X,y) 
LDA.explained_variance_ratio_ #特徵融合過後的貢獻度(0.99,0.008)
np.cumsum(LDA.explained_variance_ratio_)


