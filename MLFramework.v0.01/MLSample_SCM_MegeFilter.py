import pandas as pd
import numpy as np
from BaseClass.MLBase import MLBase ,fillNaType

class MLSample(MLBase):
    def __init__(self):
        super(MLSample, self).__init__()
        self.config.dataFiles = {
            'files':["./data/Parts_EQP_Output_ByMonth_20210407_van.csv"
                     ,"./data/ScmTrainingData_Monthly_30_20152021.csv"
                    ],
            'relations':[
                    [['MFG_MONTH','EQP_NO'],['MFG_MONTH','TOOL_ID']]
                    ,[['MFG_MONTH'],['MFG_MONTH']]
                    ]
        }
         # 初始值篩選條件
        # self.config.InputDataCondition= [
        #     {'column': 'PART_NO', 'operator': "=", 'value': '85-ECT0010'},
        # ]
        self.config.datafile = "./data/Parts_Tools_30Merge.csv"
        self.config.targetCol = "QTY"
        self.config.xAxisCol = "MFG_MONTH"
        self.config.includeColumns = []
        self.config.excludeColumns =['PM','TS','ENG','NST','STOCK_EVENT_TIME','TOOL_ID']
        self.config.fillNaType=fillNaType.MEAN  ##填補遺漏值##
        self.config.modelFileKey="Parts_Tools_30_85-ECT0010"

        self.config.forceRetrain=True

        self.config.runModel=['LRModel'] #['DNN','DNN1k','LRModel','NN','RFModel','XG']
        #self.scaler
        #self.scalerColumnList=[]

    ##資料轉換##
    def dataTransform(self):
        self.dfInputData['MFG_MONTH'] = self.dfInputData['MFG_MONTH'].astype(str)
        # self.dfInputData = self.dfInputData[self.dfInputData['PART_NO']=='85-ECT0010']


    ##特徵轉換##
    def featureTransform(self):
        self.dfInputDataRaw=  self.dfInputData.copy(deep=False)
        self.dfInputData = pd.get_dummies(self.dfInputData,columns=['EQP_NO','PART_NO'],prefix_sep='_')

    ##準備訓練資料##
    def getTrainingData(self):
        getTrainingData = self.dfInputData[(self.dfInputData['MFG_MONTH']>='201501')&(self.dfInputData['MFG_MONTH']<='202012')]
        getTrainingData=getTrainingData.drop(columns='MFG_MONTH')
        return self.dfInputData[(self.dfInputData['MFG_MONTH']>='201501')&(self.dfInputData['MFG_MONTH']<='202012')]

    ##準備測試資料##
    def getTestingDataRaw(self):
        return self.dfInputDataRaw[(self.dfInputDataRaw['MFG_MONTH']>='202101')&(self.dfInputDataRaw['MFG_MONTH']<='202103')]

    def getTestingData(self):
        return self.dfInputData[(self.dfInputData['MFG_MONTH']>='202101')&(self.dfInputData['MFG_MONTH']<='202103')]

if __name__ == "__main__":
    sample=MLSample()
    sample.run()
