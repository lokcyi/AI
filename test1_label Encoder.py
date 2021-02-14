#(一)
import pandas as pd
#label Encoder ex 天氣的資料  類別型的 資料轉換成數值型資料
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
data=pd.DataFrame([['windy','hot',25],['sunny','hot',30],['cloudy','cold',18]],columns=['weather','feeling','temperature'])
df=data.copy()
#1.手動轉換  自己製作weather mapping
weather_mapping={'windy':0,'sunny':1,'cloudy':2}
feeling_mapping={'hot':0,'cold':1}
df['weather']=df['weather'].map(weather_mapping)
df['feeling']=df['feeling'].map(feeling_mapping)

#2.label Encoder 自動轉換 
le=LabelEncoder()
df['weather']=le.fit_transform(df['weather'])
df['feeling']=le.fit_transform(df['feeling'])

#用迴圈方式轉換所有(選擇) columns
#for col in df.columns:
for col in df[['weather','feeling']]:
    df[col]=le.fit_transform(df[col])
print('OK')