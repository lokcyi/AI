import pandas as pd # 引用套件並縮寫為 pd
import matplotlib.pyplot as plt
import numpy as np

def demoDfCreate():
    # 資料準備 
    data = {
        'name': ['王小郭', '張小華', '廖丁丁', '丁小光'],
        'email': ['min@gmail.com', 'hchang@gmail.com', 'laioding@gmail.com', 'hsulight@gmail.com'],
        'grades': [60, 77, 92, 43]
    }

    # 建立 DataFrame 物件
    student_df = pd.DataFrame(data)
    print(student_df)
    student_df = student_df.append([{'name':'王大大','email':'w@gmail.com','grades':18}] , ignore_index = True)
    print(student_df)
    add_list = [ ('黃小帥', 'h@gmail.com', 19 )]
    dfNew=pd.DataFrame(add_list, columns = ['name' , 'email', 'grades'])
    student_df = student_df.append(dfNew , ignore_index = True)
    print(student_df)

def dfFromExcel():
    #load from excel
    df = pd.read_csv("TestData\\data.csv", encoding='utf-8')   
    # print(df)
    # print(df.LotID.value_counts())
  
    #資料重新處理
    df_2 = df.LotID.value_counts().sort_index().reset_index()
    df_2.columns = ['LotID', 'count']
    # print(df_2)
    # print(df_2['count'].min())
    # df_2['process'] = df_2['LotID'].apply(lambda x: int(x[-1:])) +  df_2['count']
    # print(df_2)
    #get rows
    print(df_2.loc[:3])
    print(df_2.loc[df_2['LotID']=='MNK816000'])
    #set value
    df_2.loc[df_2['LotID']=='MNK816000','count'] = 8
    print(df_2.loc[df_2['LotID']=='MNK816000'])
    print(df_2.iloc[lambda x: x['count'].values % 2 == 0])

    return df_2

def dataVisualize(df_2 , drawStyle):    
    #scatter / bar
    df_2.plot(kind=drawStyle,x='LotID',y='count',color='red',figsize=(12,6))
    plt.xticks(rotation=90,fontsize=7)
    plt.show()

if __name__ == '__main__':
    # demoDfCreate()
    df_2 = dfFromExcel()
    dataVisualize(df_2,'bar')

