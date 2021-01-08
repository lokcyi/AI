import pandas as pd
#OneHotEncoder
#順序型 類別型資料 ex滿意...不滿意...==>LabelEncoder  #將不同類別型的填上 0 1 2 3 4
#不具順序性的資料 男/女 ==>OneHotEncoder (順序沒有意義)
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from sklearn.compose import ColumnTransformer

data=pd.DataFrame([['A','windy','hot',25],['B','sunny','hot',30],['B','cloudy','cold',30]],columns=['Area','weather','feeling','temperature'])
df=data.copy()
#要先用label Encoder 轉成數值化之後 才能 用OneHotEncoder
enc=OneHotEncoder()
le=LabelEncoder()
for col in df[['Area','weather','feeling']]:
    df[col]=le.fit_transform(df[col])
df_ohe=enc.fit_transform(df).toarray()
pd.DataFrame(df_ohe)

#只要轉換weather就好了 (改版了)
#enc_1=OneHotEncoder(categories_features=[1]) #1:weaher
ct=ColumnTransformer([("weather", OneHotEncoder(), [1])], remainder = 'passthrough')
df_ohe=ct.fit_transform(df)
pd.DataFrame(df_ohe)

#get_dummies <== dataframe 可以直接轉二元特徵資料
df=data.copy()
df_dum=pd.get_dummies(df)

print('OK')
