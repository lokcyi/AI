import pandas as pd
import numpy as np
from BaseClass.MLBase import MLBase ,fillNaType
class MLSample(MLBase):
    def __init__(self):
        super(MLSample, self).__init__()
        self.config.datafile  = './data/prodkpi/prodkpi_POC_0411_PK_DUVKrF.csv'
        self.config.targetCol = 'INLINE_CT'
        self.config.xAxisCol = "Key"
        self.config.includeColumns = []
        self.config.excludeColumns = ['TC', 'PROCESS_TIME', 'NO_HOLD_WIP_HOURLY', 'MOVE_QTY', 'MOVE_QTY_INTERNAL',
                            'C_TOOLG_LOADING', 'C_TOOL_LOADING', 'DISPATCHING', 'BACKUP_BY_RATE', 'BACKUP_FOR_RATE',
                                'BATCH_SIZE','MOVE_RATIO','MOVE_RATIO_INTERNAL','INLINE_CT_BY_WAFER']
        self.config.modelFileKey="INLineCT"
        self.config.forceRetrain=True

        self.config.runModel = ['LRModel']
        self.config.fillNaType=fillNaType.MEAN

        #self.config.runModel=['LRModel','RFModel','NN']

        #self.scaler
        #self.scalerColumnList=[]

    ##資料轉換##
    def dataTransform(self):
        self.dfInputData['key'] =  self.dfInputData['MFG_DATE'].astype(str)+'_' +self.dfInputData['TOOLG_ID'].astype(str) +'_' +self.dfInputData['PROD_ID'].astype(str)
        self.dfInputData['MFG_DATE'] = self.dfInputData['MFG_DATE'].astype(str)
        self.dfInputData = self.dfInputData[(self.dfInputData['TOOLG_ID']== 'PK_DUVKrF')& (df_train_orign['PROD_ID']=='C11MD01A')]

    ##填補遺漏值##
    # def fillnull(self):
    #     self.dfInputData[self.nullColumnlist]=self.dfInputData[self.nullColumnlist].fillna(0)

    ##特徵轉換##
    def featureTransform(self):
        self.dfInputData = pd.get_dummies(self.dfInputData,columns=['TOOLG_ID','PROD_ID'],prefix_sep='_')

    ##準備訓練資料##
    def getTrainingData(self):
        return self.dfInputData[(self.dfInputData['MFG_DATE']<=  pd.to_datetime(final_date))]

    ##準備測試資料##
    def getTestingData(self):
        return self.dfInputData[(self.dfInputData['MFG_DATE'] >  pd.to_datetime(final_date))]
if __name__ == "__main__":
    sample=MLSample()
    sample.run()
