#(三) rescaling 特徵縮放
#多數模型背後的計算方式，都會因為scale的差異而給予市值因子更大的影響力
#建模前都應該要將各個特徵因子進行特徵縮放以規格化，且能夠讓我們在做梯度下降時更快的收斂
#方法1:Rescaling
#方法2:Standardization
#還有非常多種方法，EX:特徵集中在[-1,1]的normalization
#或是非線性的作法如:mapping to a uniform distribution,gaussian distribution

import pandas as pd
import numpy as np
import sklearn.preprocessing as sp
data1=np.random.randint(0,100,size=50)
data2=np.random.randint(100,1000,size=50)
df=pd.DataFrame([data1,data2])
df=df.T
df.head(10)
df.describe()
print('OK')
# rescaling 特徵縮放(0~1) 特徵最小/全距
FS_1= sp.MinMaxScaler().fit(df)
result_minmax= FS_1.transform(df)
#Standardization
FS_2=sp.StandardScaler().fit(df)
result_std=FS_2.transform(df)
#column 0 平均值
result_std[:,0].mean()
#column 1 平均值
result_std[:,1].mean()
#column 0 標準差
result_std[:,0].std()
#column 1 標準差
result_std[:,1].std()