# 用機台增量資料~~產生到2015年的數據

from os import replace,path
import pandas as pd
import numpy as np
from BaseClass.MLBase import MLBase ,fillNaType

class MLSample(MLBase):
    def __init__(self):
        super(MLSample, self).__init__()
        self.log.debug('{}-------------'.format(path.basename(__file__)))
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
        self.config.datafile = "./data/parts20210420/dataset_monthly_20210419.csv"
        self.config.targetCol = "QTY"
        self.config.xAxisCol = "MFG_MONTH"
        self.config.includeColumns = []
        self.config.excludeColumns = ['STOCK_EVENT_TIME','AVG', 'PM', 'TS', 'ENG', 'NST', 'TOOL_ID', 'KEY', 'BACKUP_BY_RATE', 'BACKUP_FOR_RATE', 'RWORK_LOT_RATE', 'SAMPLING_RATE', 'SHIFT1', 'SHIFT2', 'SHIFT3', 'ENG']  #,'CHANGE_RECIPE'
        self.config.encoderColumns =['PART_NO','EQP_NO'] #vanessa
        self.config.fillNaType=fillNaType.MEAN
        self.config.modelFileKey="Parts_Tools_30Quater_85-EKA0190"
        self.config.forceRetrain = True
        # 初始值篩選條件
        self.config.InputDataCondition= [
            {'column': 'PART_NO', 'operator': "=", 'value': '85-EKA0190'},
        ]
        # 訓練集篩選條件
        self.config.TrainCondition = [
            {'column':"MFG_MONTH",'operator':">=",'value': '201501'},
            {'column':"MFG_MONTH",'operator':"<=",'value':'202012'},
        ]
        # 測試集篩選條件
        self.config.TestCondition = [
            {'column':"MFG_MONTH",'operator':">",'value':'202012'},
        ]

        self.config.runModel=['NN']# ['DNN','DNN1k','LRModel','NN','RFModel','XG']
        #self.scaler
        #self.scalerColumnList=[]
        self.dataPreHandler()
    ##資料合併##
    def dataPreHandler(self):
        pass
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

    #partList =['85-ECT0010','85-EKA0190','85-EKA0270' ,'85-EMA0900','85-EMA0910','85-EMA0920', '87-WPT1070']
    partList =['85-ECT0010']
    for p in partList:
        sample.config.modelFileKey="Parts_Tools_30_Quater_{}".format(p)
        sample.config.InputDataCondition[0]['value'] = p
        sample.run()


