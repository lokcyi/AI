import pandas as pd
import numpy as np 
from BaseClass.MLBase import MLBase 
class MLSample(MLBase):
    def __init__(self):
        super(MLSample, self).__init__()
        self.config.datafile = "./data/m4.csv" 
        self.config.targetCol = "INLINE_CT_BY_LOT"
        self.config.xAxisCol = "MFG_DATE"
        self.config.includeColumns = []
        self.config.excludeColumns =['MOVE_QTY','INLINE_CT_BY_WAFER','RUN_WIP_RATIO']
        self.config.modelFileKey="INLINE_CT_BY_LOT" 
        self.config.forceRetrain=False 
        #self.config.runModel=['LRModel','RFModel','NN']
        self.config.runModel=['LRModel','RFModel','XG','DNN','NN']

        #self.scaler
        #self.scalerColumnList=[]

    ##資料轉換##    
    def dataTransform(self):
        self.dfInputData['MFG_DATE'] = self.dfInputData['MFG_DATE'].astype(str)   
        self.dfInputData = self.dfInputData[self.dfInputData['TOOLG_ID']=='PK_DUVKrF']  
        #self.dfInputData['HoldWIP']=self.dfInputData['WIP_QTY']-self.dfInputData['NO_HOLD_QTY']
        #self.dfInputData['HoldWIP_AVG']=self.dfInputData.HoldWIP.rolling(10).sum()-self.dfInputData['HoldWIP']
        #self.dfInputData['HoldWIP_AVG1']=self.dfInputData.HoldWIP.rolling(3).sum()-self.dfInputData['HoldWIP']
        #self.dfInputData['WIP_QTY_AVG']=self.dfInputData.WIP_QTY.rolling(3).sum()-self.dfInputData['WIP_QTY']
        

    ##填補遺漏值##
    def fillnull(self):
        self.dfInputData[self.nullColumnlist]=self.dfInputData[self.nullColumnlist].fillna(0) 
    
    ##特徵轉換##
    def featureTransform(self):        
        self.dfInputData = pd.get_dummies(self.dfInputData,columns=['TOOLG_ID'],prefix_sep='_')  

    ##準備訓練資料##
    def getTrainingData(self):
        return self.dfInputData[(self.dfInputData['MFG_DATE']>='20200101')&(self.dfInputData['MFG_DATE']<='20210211')]         

    ##準備測試資料##
    def getTestingData(self):
        return self.dfInputData[(self.dfInputData['MFG_DATE']>='20210212')&(self.dfInputData['MFG_DATE']<='20210228')]     

sample=MLSample()
sample.run()
 