import pandas as pd
import numpy as np 
from enum import Enum
from BaseClass.MLBase import MLBase 
class fillNaType(Enum):
    MEAN = 'mean'
    BFILL = 'bfill'
    FFILL = 'ffill'
    DROPNA = 'dropna'
    ZERRO = 'zero'
    MODE = 'mode'
class MLSample(MLBase):
    def __init__(self):
        super(MLSample, self).__init__()
        self.config.datafile = "./data/Parts_Tools_30.csv" 
        self.config.targetCol = "QTY"
        self.config.xAxisCol = "MFG_MONTH"
        self.config.includeColumns = []
        self.config.excludeColumns =['PM','TS','Tool.MFG_MONTH','Tool.TOOL_ID'
        ]
        self.config.fillNaType=fillNaType.MEAN
        self.config.modelFileKey="Parts_Tools_30" 
        self.config.forceRetrain=False 
         
        # self.config.runModel=['DNN','DNN1k','LRModel','NN','RFModel','XG']
        self.config.runModel=['LRModel','CAT','NN']
        #self.config.runModel=['CAT']
        #self.scaler
        #self.scalerColumnList=[]

    ##資料轉換##    
    def dataTransform(self):
        self.dfInputData['MFG_MONTH'] = self.dfInputData['MFG_MONTH'].astype(str)   
        self.dfInputData = self.dfInputData[self.dfInputData['PART_NO']=='85-ECT0010']  

    ##填補遺漏值##
    def fillnull(self):
        if(self.config.fillNaType.value=='mean'): 
            self.dfInputData[self.nullColumnlist] = self.dfInputData[self.nullColumnlist].fillna(self.dfInputData.median()).fillna(value=0)
        elif(fillNaType.value=='mode'):  
            self.dfInputData = self.dfInputData.fillna(self.dfInputData.mode())            
        elif(fillNaType.value=='bfill'):  
            self.dfInputData = self.dfInputData.fillna(method='bfill').fillna(self.dfInputData.median())
        elif(fillNaType.value=='ffill'):  
            self.dfInputData = self.dfInputData.fillna(method='ffill').fillna(self.dfInputData.median())
        elif(fillNaType.value=='dropna'): 
            self.dfInputData = self.dfInputData.dropna()
        elif(fillNaType.value=='zero'):   
            self.dfInputData[self.nullColumnlist]=self.dfInputData[self.nullColumnlist].fillna(0) 
    
    ##特徵轉換##
    def featureTransform(self):      
        self.dfInputDataOrg=  self.dfInputData.copy(deep=False)
        self.dfInputData = pd.get_dummies(self.dfInputData,columns=['EQP_NO','PART_NO'],prefix_sep='_')  

    ##準備訓練資料##
    def getTrainingData(self):
        return self.dfInputData[(self.dfInputData['MFG_MONTH']>='202002')&(self.dfInputData['MFG_MONTH']<='202012')]         

    ##準備測試資料##
    def getTestingDataOrg(self):
        return self.dfInputDataOrg[(self.dfInputDataOrg['MFG_MONTH']>='202101')&(self.dfInputDataOrg['MFG_MONTH']<='202103')]  
    def getTestingData(self):
        return self.dfInputData[(self.dfInputData['MFG_MONTH']>='202101')&(self.dfInputData['MFG_MONTH']<='202103')]     

sample=MLSample()
sample.run()
