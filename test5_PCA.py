#(五) PCA降維(Principal Component Analysis) 主成分分析
#PCA是非監督式學習 
#透過線性轉換，降低原始特徵的維度，並盡可能的保留原始特徵的差異性
#相似度高的因子 合併為一個因子
#透過 映射(projection)的方式
#PCA是反過來將特從高維度映射到低維度，並讓這些特徵的變異越大越好(SVN是透過低維度映射到高維度)
#無監督式
#PCA就是一個最佳化的問題，旨在找到讓映射後的資料變異量最大的投影向量
#優點:是可以盡可能地在訊息損失及少的情況下，透過降維的方式減低數據量
#缺點:在映射的過程中，透過特徵的融合，最後要解釋模型的時候會比較困難，特徵轉換過後非原本純粹的特徵

from sklearn.decomposition import PCA
from sklearn import datasets
import pandas as pd
import numpy as np

iris=datasets.load_iris()
X=iris.data #鳶尾花四個特徵資料
y=iris.target # 三種不同類型的鳶尾花

pca=PCA(n_components=2) #n_components 是最終的維度 
pca.fit(X).transform(X) #原本四個特徵 變成兩個特徵
pca.n_components_ #output 2 特徵
pca.explained_variance_ratio_ #每個component 的貢獻度==>output (0.92,0.053)
np.cumsum(pca.explained_variance_ratio_) #計算 PCA 降維後 保留了多少的特徵貢獻度 ==>output(0.92,0.97)

#結論:PCA降維 建議所有的特徵都要經過標準化在近來處理，並免因位scale的關係導致映射失準




