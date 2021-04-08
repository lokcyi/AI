import pandas as pd
import os
import numpy as np 
import joblib as joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler #平均&變異數標準化 平均值為0，方差為1。
from sklearn.preprocessing import MinMaxScaler #最小最大值標準化[0,1]
from sklearn.preprocessing import RobustScaler #中位數和四分位數標準化
from sklearn.preprocessing import MaxAbsScaler #絕對值最大標準化
 
 

class Data:    
    @staticmethod
    def readData(inputfile):
        df = pd.read_csv(inputfile)
        df=df.dropna(axis=1,how='all')
        df.info() 
        return Data.analyzeData(df) 
    @staticmethod
    def analyzeData(df):
        print('非數值欄位：')
        strColumnlist=df.select_dtypes(exclude=['int64','float64']).columns.tolist()
        print(strColumnlist) 
        print('數值欄位：')
        numbericColumnlist=df.select_dtypes(include=['int64','float64']).columns.tolist()
        print(numbericColumnlist)
        print('包含ＮＵＬＬ的欄位：')
        nullColumnlist=df.columns[df.isna().any()].tolist()    
        print(nullColumnlist)
        print('===================================================')  
        return df,strColumnlist,numbericColumnlist,nullColumnlist
    @staticmethod
    def filterColumns(df,config):
        includeColumns=config.includeColumns
        excludeColumns=config.excludeColumns
        if (len(includeColumns)>0):
            df=df[includeColumns]
        df=df.drop(columns=excludeColumns)
        return df
    @staticmethod
    def scalerData(df,scalerKind,numbericColumnlist,config):
        target_cols=config.targetCol
        scalerColumnlist = [ele for ele in numbericColumnlist if ele not in target_cols]
        scaler = MinMaxScaler()
        scaler.fit(df[scalerColumnlist])
        df[scalerColumnlist]= scaler.transform(df[scalerColumnlist])    
        return df

    @staticmethod
    def accsum(def_result,target_cols):
        _accsum=0 
        for index,row in def_result.iterrows():
            #避免當分母為0 會無法計算
            if row[target_cols] <0 :
                row[target_cols]  =0.00001
            if row[target_cols] ==0.0:
                row[target_cols]  =0.00001
            if row['Predict'] <0 :
                row['Predict']  =0 
            if 1- abs((row['Predict'] - row[target_cols])/row[target_cols] ) >0 : 
                _accsum+=(1- abs((row['Predict'] - row[target_cols])/row[target_cols] ))
        
        return round(_accsum*100/def_result.shape[0],2)
    @staticmethod
    def accsumAggrigation(def_result,target_cols,groupbyCols):
        _accsum=0    
        def_result_summary = def_result.groupby(groupbyCols, as_index=False)[target_cols,'Predict'].sum()
        
        def_result_summary[def_result_summary[target_cols] ==0.0][target_cols]  =0.0001
        def_result_summary[def_result_summary['Predict'] <0]['Predict']  =0
        for index,row in def_result_summary.iterrows():
            if row[target_cols] <0 :
                row[target_cols]  =0.00001
            if row[target_cols] ==0.0:
                row[target_cols]  =0.00001
            if 1- abs((row['Predict'] - row[target_cols])/row[target_cols] ) >0 :
                _accsum+=(1- abs((row['Predict'] - row[target_cols])/row[target_cols] ))
        
        return round(_accsum*100/def_result_summary.shape[0],2)

 
    @staticmethod
    def testModel(XTest,model,mlKind,df,config):
        yTest=model.predict(XTest)
        df2=df.copy(deep=False)
        df2.insert(len(df2.columns), 'Predict', yTest)     
        # plt.figure(figsize=(20,6))
        plt.title((mlKind+":{0}%  {1}%").format(Data.accsum(df2,config.targetCol),Data.accsumAggrigation(df2,config.targetCol,['PART_NO','MFG_MONTH'])))    
        plt.xlabel(config.xAxisCol)
        plt.xticks(rotation=90)        
        plt.ylabel(config.targetCol)
        t = df2[config.xAxisCol].to_numpy()+'_'+np.arange(len(XTest)).astype(str)  # 创建t变量
        # plt.plot(df2[config.xAxisCol],df2['Predict'], label = mlKind, color='red', marker='.',linewidth = '0.5')
        # plt.plot(df2[config.xAxisCol],df2[config.targetCol], label = "ACT", color='blue', marker='.',linewidth = '0.5')
        plt.plot(t,df2['Predict'], label = mlKind, color='red', marker='.',linewidth = '0.5')
        plt.plot(t,df2[config.targetCol], label = "ACT", color='blue', marker='x',linewidth = '0')
        plt.legend()
        plt.ylim(bottom=0) 
        
        df2.to_csv('./Report/'+config.modelFileKey+'_'+mlKind+'.csv',index=False)
        print("Test acc%:",mlKind,Data.accsum(df2,config.targetCol)) 
        print("Test acc Aggreation%:",mlKind,Data.accsumAggrigation(df2,config.targetCol,['PART_NO','MFG_MONTH'])) 


# if __name__ == "__main__": 
    
#     data.testModel()
#     data.readData('./Report/'+config.modelFileKey+'_'+mlKind+'.csv')