
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
#print (data)

#print (data.describe(include='all'))

#print (data.head())

#print (data.describe())

print (data.Embarked.value_counts()) 


#Embarked是乘客上船的港口，S是Embarked最常出現的類別，代表最多人上船的港口。
#由於可以計數，可以使用長條圖與圓餅圖 做視覺化，例如長條圖與圓餅圖。

#長條圖
#data.Embarked.value_counts().plot.barh(x='Port of Embarkation', y='number of people')
#plt.show()

#圓餅圖
#data.Embarked.value_counts().plot(kind='pie')
#plt.show()

