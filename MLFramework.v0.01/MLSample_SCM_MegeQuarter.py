from os import replace
import pandas as pd
import numpy as np 
from BaseClass.MLBase import MLBase ,fillNaType

class MLSample(MLBase):
    def __init__(self):
        super(MLSample, self).__init__()
        # self.config.dataFiles = {
        #     'files':["./data/Parts_EQP_Output_ByMonth_20210407_van.csv"
        #              ,"./data/ScmTrainingData_Monthly_30_20152021.csv"
        #              # ,"./data/ScmTrainingData_Monthly_30days.csv"
        #             ,"./data/holiday.csv"
        #             ],
        #     'relations':[
        #             [['MFG_MONTH','EQP_NO'],['MFG_MONTH','TOOL_ID']]
        #             ,[['MFG_MONTH'],['MFG_MONTH']]
        #             ]
        # } 
        self.config.datafile = "./data/Parts_Tools_30Quater.csv" 
        self.config.targetCol = "QTY"
        self.config.xAxisCol = "MFG_MONTH"
        self.config.aggreationCol = ['PART_NO','MFG_MONTH']
        self.config.includeColumns = []
        self.config.excludeColumns =['PM','TS','ENG','NST','STOCK_EVENT_TIME','TOOL_ID']
        self.config.fillNaType=fillNaType.MEAN
        self.config.modelFileKey="Parts_Tools_30Quater_85-EMA0130" 
        self.config.forceRetrain=True
         
        # self.config.runModel=['DNN','DNN1k','LRModel','NN','RFModel','XG']
        self.config.runModel=['LRModel','NN','CAT']
        self.partno='85-EMA0130'
# 85-EMA0920
# 85-EMA0130
# 85-EMA0910 怪
# 85-ECT0010 good
# 85-EMA0900 NG
# 86-DIA0120 OK
# 87-WPT1070 soso
# 85-EKA0270 not bad
# 85-EKA0190 goodß
        # self.config.runModel=['CAT']
        #self.scaler
        #self.scalerColumnList=[]
        self.dataPreHandler()
    ##資料轉換##    
    def dataPreHandler(self):
        df_parts=pd.read_csv("./data/Parts_EQP_Output_ByMonth_20210407_van.csv")
        df_parts['MFG_MONTH'] = pd.to_datetime(df_parts['STOCK_EVENT_TIME'].values, format='%Y-%m-%d').astype('period[Q]')
        df_parts.drop(columns=['STOCK_EVENT_TIME'],inplace=True)
        df_parts = df_parts.groupby(['PART_NO','EQP_NO','MFG_MONTH']).sum().reset_index()
        df_EQP=pd.read_csv("./data/ScmTrainingData_Monthly_30_20152021.csv")
        df_EQP['MFG_MONTH'] = pd.to_datetime(df_EQP['MFG_MONTH'].values, format='%Y%m').astype('period[Q]')
        df_EQP = df_EQP.groupby(['TOOL_ID','MFG_MONTH']).mean().reset_index()
        df_merge = pd.merge(df_parts, df_EQP, left_on=['EQP_NO','MFG_MONTH'], right_on=['TOOL_ID','MFG_MONTH'],how="inner")
        df_merge.to_csv(self.config.datafile, index=False)

        

    def dataTransform(self):
        self.dfInputData['MFG_MONTH'] = self.dfInputData['MFG_MONTH'].astype(str)   
        self.dfInputData = self.dfInputData[self.dfInputData['PART_NO']==self.partno]  

    ##填補遺漏值##
    def fillnull(self):
        if(self.config.fillNaType.value=='mean'): 
            self.dfInputData[self.nullColumnlist] = self.dfInputData[self.nullColumnlist].fillna(self.dfInputData.median()).fillna(value=0)
        elif(self.fillNaType.value=='mode'):  
            self.dfInputData = self.dfInputData.fillna(self.dfInputData.mode())            
        elif(self.fillNaType.value=='bfill'):  
            self.dfInputData = self.dfInputData.fillna(method='bfill').fillna(self.dfInputData.median())
        elif(self.fillNaType.value=='ffill'):  
            self.dfInputData = self.dfInputData.fillna(method='ffill').fillna(self.dfInputData.median())
        elif(self.fillNaType.value=='dropna'): 
            self.dfInputData = self.dfInputData.dropna()
        elif(self.fillNaType.value=='zero'):   
            self.dfInputData[self.nullColumnlist]=self.dfInputData[self.nullColumnlist].fillna(0) 
    
    ##特徵轉換##
    def featureTransform(self):      
        self.dfInputDataRaw=  self.dfInputData.copy(deep=False)
        self.dfInputData = pd.get_dummies(self.dfInputData,columns=['EQP_NO','PART_NO'],prefix_sep='_')  

    ##準備訓練資料##
    def getTrainingData(self):
        getTrainingData = self.dfInputData[(self.dfInputData['MFG_MONTH']>='2015Q1')&(self.dfInputData['MFG_MONTH']<='2021Q1')]  
        getTrainingData=getTrainingData.drop(columns='MFG_MONTH')
        return self.dfInputData[(self.dfInputData['MFG_MONTH']>='2015Q1')&(self.dfInputData['MFG_MONTH']<='2021Q1')]         

    ##準備測試資料##
    def getTestingDataRaw(self):
        return self.dfInputDataRaw[(self.dfInputDataOrg['MFG_MONTH']>='2021Q1')&(self.dfInputDataOrg['MFG_MONTH']<='2021Q1')]  

    def getTestingData(self):
        return self.dfInputData[(self.dfInputData['MFG_MONTH']>='2021Q1')&(self.dfInputData['MFG_MONTH']<='2021Q1')]     

if __name__ == "__main__": 
    sample=MLSample()
    sample.run()
