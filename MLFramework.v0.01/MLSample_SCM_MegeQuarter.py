# 用機台增量資料~~產生到2015年的數據

from os import replace,path
import pandas as pd
import numpy as np
from BaseClass.MLBase import MLBase ,fillNaType

class MLSample(MLBase):
   def __init__(self):
        super(MLSample, self).__init__()
        self.log.debug('-------------{}-------------'.format(path.basename(__file__)))
      #   self.config.dataFiles = {
      #       'files':["./data/Parts_EQP_Output_ByMonth_20210407_van.csv"
      #                ,"./data/ScmTrainingData_Monthly_30_20152021.csv"
      #                # ,"./data/ScmTrainingData_Monthly_30days.csv"
      #               ,"./data/holiday.csv"
      #               ],
      #       'relations':[
      #               [['MFG_MONTH','EQP_NO'],['MFG_MONTH','TOOL_ID']]
      #               ,[['MFG_MONTH'],['MFG_MONTH']]
      #               ]
      #   }
        self.config.datafile = "./data/Parts_Tools_30Quater.csv"
        self.config.targetCol = "QTY"
        self.config.xAxisCol = "MFG_MONTH"
        self.config.includeColumns = []
        self.config.excludeColumns = ['PM','TS','ENG','NST']# ,'TOOL_ID','BACKUP_BY_RATE','SAMPLING_RATE','NUM_RECIPE','CHANGE_RECIPE']#,]
        self.config.encoderColumns =['PART_NO','EQP_NO','TOOL_ID'] #vanessa
        self.config.fillNaType=fillNaType.MEAN
        self.config.modelFileKey="Parts_Tools_30Quater_85-EKA0190"
        self.config.forceRetrain = True
        # 初始值篩選條件
        self.config.InputDataCondition= [
            {'column': 'PART_NO', 'operator': "=", 'value': '85-EKA0190'},
        ]
        # 訓練集篩選條件
        self.config.TrainCondition = [
            # {'column':"MFG_MONTH",'operator':">=",'value': '2015Q1'},
            {'column':"MFG_MONTH",'operator':"<=",'value':'2020Q4'},
        ]
        # 測試集篩選條件
        self.config.TestCondition = [
            {'column':"MFG_MONTH",'operator':">",'value':'2020Q4'},
        ]

        self.config.runModel=['CAT','XG','LRModel','NN','RFModel']# ['DNN','DNN1k','LRModel','NN','RFModel','XG']
        #self.scaler
        #self.scalerColumnList=[]
        self.dataPreHandler()
   # ##資料合併##

   def dataPreHandler(self):
      #   pass
        df_parts=pd.read_csv("./data/Parts_EQP_Output_ByMonth_20210407_van.csv")
        df_parts['MFG_MONTH'] = pd.to_datetime(df_parts['STOCK_EVENT_TIME'].values, format='%Y-%m-%d').astype('period[Q]')
        df_parts.drop(columns=['STOCK_EVENT_TIME'],inplace=True)
        df_parts = df_parts.groupby(['PART_NO','EQP_NO','MFG_MONTH']).sum().reset_index()
        df_EQP=pd.read_csv("./data/ScmTrainingData_Monthly_30_202002.csv")
        df_EQP['MFG_MONTH'] = pd.to_datetime(df_EQP['MFG_MONTH'].values, format='%Y%m').astype('period[Q]')
        df_EQP = df_EQP.groupby(['TOOL_ID','MFG_MONTH']).mean().reset_index()
        df_merge = pd.merge(df_parts, df_EQP, left_on=['EQP_NO','MFG_MONTH'], right_on=['TOOL_ID','MFG_MONTH'],how="inner")
        df_merge.to_csv(self.config.datafile, index=False)


    ##資料轉換##
   def dataTransform(self):
      self.dfInputData['MFG_MONTH'] = self.dfInputData['MFG_MONTH'].astype(str)

    # ##填補遺漏值##
    # def fillnull(self):
    #     if(self.config.fillNaType.value=='mean'):
    #         self.dfInputData[self.nullColumnlist] = self.dfInputData[self.nullColumnlist].fillna(self.dfInputData.median()).fillna(value=0)
    #     elif(self.fillNaType.value=='mode'):
    #         self.dfInputData = self.dfInputData.fillna(self.dfInputData.mode())
    #     elif(self.fillNaType.value=='bfill'):
    #         self.dfInputData = self.dfInputData.fillna(method='bfill').fillna(self.dfInputData.median())
    #     elif(self.fillNaType.value=='ffill'):
    #         self.dfInputData = self.dfInputData.fillna(method='ffill').fillna(self.dfInputData.median())
    #     elif(self.fillNaType.value=='dropna'):
    #         self.dfInputData = self.dfInputData.dropna()
    #     elif(self.fillNaType.value=='zero'):
    #         self.dfInputData[self.nullColumnlist]=self.dfInputData[self.nullColumnlist].fillna(0)

    ##特徵轉換##
   def featureTransform(self):
        self.dfInputDataRaw=  self.dfInputData.copy(deep=False)
        self.dfInputData = pd.get_dummies(self.dfInputData,columns=self.config.encoderColumns,prefix_sep='_') #vanessa

    ##準備訓練資料##
   def getTrainingData(self):
        df = self.dfInputData.copy(deep=False)
        for c in self.config.TrainCondition  :
            if c['operator'] == "=":
               df =df[df[c['column']] == c['value']]
            elif c['operator']  == "=!":
               df =df[df[c['column']] != c['value']]
            elif c['operator']  == "<=":
               df =df[df[c['column']] <= c['value']]
            elif c['operator']  == "<":
               df =df[df[c['column']] < c['value']]
            elif  c['operator']  == ">=":
               df =df[df[c['column']] >=c['value']]
            elif c['operator']  ==  ">":
               df = df[df[c['column']] > c['value']]

        df.to_csv('./log/trainingDATA_{}.csv'.format(self.config.modelFileKey))
        return  df

    ##準備測試資料##
   def getTestingDataRaw(self):
         #super(MLSample, self).getTestingDataRaw() #calling method of parent class
        df = self.dfInputDataRaw.copy(deep=False)
        for c in self.config.TestCondition  :
            if c['operator'] == "=":
               df =df[df[c['column']] == c['value']]
            elif c['operator']  == "=!":
               df =df[df[c['column']] != c['value']]
            elif c['operator']  == "<=":
               df =df[df[c['column']] <= c['value']]
            elif c['operator']  == "<":
               df =df[df[c['column']] < c['value']]
            elif  c['operator']  == ">=":
               df =df[df[c['column']] >=c['value']]
            elif c['operator']  ==  ">":
               df = df[df[c['column']] > c['value']]
        return df

   def getTestingData(self):
        df = self.dfInputData.copy(deep=False)
        for c in self.config.TestCondition  :
            if c['operator'] == "=":
               df =df[df[c['column']] == c['value']]
            elif c['operator']  == "=!":
               df =df[df[c['column']] != c['value']]
            elif c['operator']  == "<=":
               df =df[df[c['column']] <= c['value']]
            elif c['operator']  == "<":
               df =df[df[c['column']] < c['value']]
            elif  c['operator']  == ">=":
               df =df[df[c['column']] >=c['value']]
            elif c['operator']  ==  ">":
               df = df[df[c['column']] > c['value']]
        df.to_csv('./log/testingDATA_{}.csv'.format(self.config.modelFileKey))
        return  df
if __name__ == "__main__":
   sample=MLSample()

   # partList =['85-ECT0010','85-EKA0190','85-EKA0270' ,'85-EMA0900','85-EMA0910','85-EMA0920', '86-DIA0120','87-WPT1070','85-EMA0130']
   partList =['86-DIA0120']
   sample.config.runModel=['LRModel']
   for p in partList:
        sample.config.modelFileKey="Parts_Tools_30_Quater2020_{}".format(p)
        sample.config.InputDataCondition[0]['value'] = p
        sample.run()


