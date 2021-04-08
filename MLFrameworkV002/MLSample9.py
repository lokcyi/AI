import pandas as pd
import numpy as np 
from BaseClass.MLBase import MLBase 
class MLSample9(MLBase):
    def __init__(self):
        super(MLSample9, self).__init__()
        self.config.datafile = "./data/m4.csv" 
        self.config.targetCol = "INLINE_CT_BY_LOT"
        self.config.xAxisCol = "MFG_DATE"
        self.config.includeColumns = ['TOOLG_ID','MFG_DATE','INLINE_CT_BY_LOT','WIP_QTY','C_CLOSE_WIP','NO_HOLD_QTY','PROCESS_JOBTIME',
        'C_EQP_UTIL','EQP_UTIL','BACKUP_FOR_RATE','RUN_WIP','QUE_LOT_RATE']
        self.config.excludeColumns =[]
        self.config.modelFileKey="555" 
        self.config.forceRetrain=False 
        #self.config.runModel=['LR','XG','RF','DNN','NN']
        self.config.runModel=['LRModel','RFModel','XG','DNN','NN']

        #self.scaler
        #self.scalerColumnList=[]
       

    ##資料轉換##    
    def dataTransform(self):
        self.dfInputData['MFG_DATE'] = self.dfInputData['MFG_DATE'].astype(str)   
        self.dfInputData = self.dfInputData[self.dfInputData['TOOLG_ID']=='PK_DUVKrF']  

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

sample=MLSample9()
sample.run()
 