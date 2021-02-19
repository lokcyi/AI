
import pandas as pd
#匯入maplotlib做視覺化
import matplotlib.pyplot as plt

#讀取資料
#data = pd.read_csv('data/train.csv')
#print (data)

#titanic的統計敘述

column_types={'PassengerId':'category',
               'Survived':'category',
                'Pclass':int,
                'Name':'category',
                'Sex':'category',
                'Age':float,
                'SibSp':int,
                'Parch':int,
                'Fare':float,
                'Cabin':'category',
                'Embarked':'category',}
data = pd.read_csv('data/train.csv', dtype=column_types)

#print (data.describe())
#print (data)

from pandas.plotting import scatter_matrix
scatter_matrix(data, alpha=0.5, diagonal='kde')
plt.show()
