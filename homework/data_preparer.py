import numpy as np  
import pandas as pd  
  

def preprocess(data_df, sRatio=None):  
    # Show top 2 records  
    #print("\n[Info] Show top 2 records:")  
    #print(data_df.values[:2])

    # 訓練時不需 "姓名"，移除此欄位 (預測階段才會用到)
    # if ['Name'] in data_df.columns:
    #     data_df = data_df.drop(['Name'], axis=1)  
    data_df =data_df.drop(['IDX','0I', '1C', '1D', '1F', '1G', '1I', '1M', '1N', '1P',
       '1T', '1U', '1V', '2D', '2E', '2F', '2I', '2M', '2N', '2P', '2T', '2U',
       '2V', '3D', '3E', '3G', '3I', '3K', '3M', '3N', '3P', '3S', '3T', '3U',
       '3V', '4D', '4E', '4I', '4M', '4N', '4T', '4U', '4V', '5D', '5E', '5I',
       '5M', '5N', '5P', '5S', '6D', '6I', '6N', '6P', '7D', '7I', '8D', '8I',
       '9D', 'DO', 'ES', 'GN', 'HR', 'LI', 'PO', 'PV', 'SP', 'SV', 'TM', 'TU',
       'TV', 'UG'], axis=1)  
       
    # 含 null 值的資料筆數
    #print("\n[Info] 含 null 值的資料筆數:")  
    #print(data_df.isnull().sum()) 
    
    # 若 "年齡" 為 null，填入平均值
    #print("\n[Info] 處理 "年齡" 為 null 的資料")  
    # if 'Age' in data_df.columns:
    #     age_mean = data_df['Age'].mean()  
    #     data_df['Age'] = data_df['Age'].fillna(age_mean)  
    
    # 將 "性別" 轉換成 0, 1
    #print("\n[Info] 將 "性別" 轉換成數值")  
    if 'Sex' in data_df.columns:        
        data_df['Sex'] = data_df['Sex'].map({'female':0, 'male':1}).astype(int)  

    # 將 "登船港口" 以 Onehot Encoding 進行轉換
    #print("\n[Info] 將 "登船港口" 以 Onehot Encoding 進行轉換")  
    if 'Embarked' in data_df.columns:        
        data_df = pd.get_dummies(data=data_df, columns=['Embarked'])  
    




    # 將 dataframe 轉換為 array
    ndarray = data_df.values  
    #print("\n[Info] Translate into ndarray(%s) with shape=%s" % (ndarray.__class__, str(ndarray.shape)))  
    #print("\n[Info] Show top 2 records:\n%s\n" % (ndarray[:2]))  
    #print("\n[Info] ndarray:")
    #print(ndarray)

    # Separate labels with features  
    Label = ndarray[:,0]  
    Features = ndarray[:,1:]  

    #print("\n[Info] Label:")
    #print(Label)
    #print("\n[Info] Features:")
    #print(Features)





    # 將特徵欄位進行標準化 
    #print("\n[Info] Normalized features...")  
    from sklearn import preprocessing  
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))  
    scaledFeatures = minmax_scale.fit_transform(Features)  
    #print("\n[Info] Show top 2 records:\n%s\n" % (scaledFeatures[:2])) 





    if sRatio:  
        # 切分資料為訓練資料與測試資料 
        print("\n[Info] 切分資料為訓練資料與測試資料 ")  
        msk = np.random.rand(len(scaledFeatures)) < sRatio  
        trainFeatures = scaledFeatures[msk]  
        trainLabels = Label[msk]  
        testFeatures = scaledFeatures[~msk]  
        testLabels = Label[~msk]  
        print("\t[Info] Total %d training instances; %d testing instances" % (trainFeatures.shape[0], testFeatures.shape[0]))  
        return (trainFeatures, trainLabels, testFeatures, testLabels)  
    else:  
        return (scaledFeatures, Label)  