import pandas as pd
import numpy as np
from BaseClass.MLBase import MLBase
class MLSample(MLBase):
    def __init__(self):
        super(MLSample, self).__init__()
        self.config.datafile = "./data/Parts_Tools_60.csv"
        self.config.targetCol = "QTY"
        self.config.xAxisCol = "MFG_MONTH"
        self.config.includeColumns = []
        self.config.excludeColumns =['PM','TS','Tool.MFG_MONTH','Tool.TOOL_ID'
        ]
        self.config.modelFileKey="Parts_Tools_60"
        self.config.forceRetrain=True

        self.config.runModel=['LRModel']
        #self.config.runModel=['LRModel','RFModel','NN']

        #self.scaler
        #self.scalerColumnList=[]

    ##資料轉換##
    def dataTransform(self):
        self.dfInputData['MFG_MONTH'] = self.dfInputData['MFG_MONTH'].astype(str)
        self.dfInputData = self.dfInputData[self.dfInputData['PART_NO']=='85-ECT0010']

    ##填補遺漏值##
    def fillnull(self):
        self.dfInputData[self.nullColumnlist]=self.dfInputData[self.nullColumnlist].fillna(0)

    ##特徵轉換##
    def featureTransform(self):
        self.dfInputData = pd.get_dummies(self.dfInputData,columns=['EQP_NO','PART_NO'],prefix_sep='_')

    ##準備訓練資料##
    def getTrainingData(self):
        return self.dfInputData[(self.dfInputData['MFG_MONTH']>='202002')&(self.dfInputData['MFG_MONTH']<='202012')]

    ##準備測試資料##
    def getTestingData(self):
        return self.dfInputData[(self.dfInputData['MFG_MONTH']>='202101')&(self.dfInputData['MFG_MONTH']<='202103')]

sample=MLSample()
sample.run()
