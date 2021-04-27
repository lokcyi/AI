import pandas as pd
import numpy as np
from BaseClass.MLBase import MLBase ,fillNaType
class MLSample(MLBase):
    def __init__(self):
        super(MLSample, self).__init__()
        '''
        剛剛跟文彥討論了一下，Product move ratio, Inline Cycle time 的 model 先以 Product + Toolg 為主先進行
        產品可以考慮用 C11MD01A, L80GA03A, D25AS01A
        這分別是三種產品，數量在 8000 以上
        '''
        self.config.datafile  = './data/prodkpi/prodkpi_POC_0411_PK_DUVKrF.csv'
        self.config.targetCol = 'INLINE_CT'
        self.config.xAxisCol = "key"
        self.config.includeColumns = []
        self.config.excludeColumns = ['MFG_DATE','TC', 'PROCESS_TIME', 'NO_HOLD_WIP_HOURLY', 'MOVE_QTY', 'MOVE_QTY_INTERNAL',
                             'BACKUP_BY_RATE', 'BACKUP_FOR_RATE',
                                'MOVE_RATIO','MOVE_RATIO_INTERNAL','INLINE_CT_BY_WAFER']
                               # 'C_TOOLG_LOADING' 'C_TOOL_LOADING' 'DISPATCHING' 'BATCH_SIZE'
        self.config.encoderColumns =['PROD_ID','TOOLG_ID'] #vanessa
        self.config.modelFileKey="INLineCT"
        self.config.forceRetrain=True

        self.config.runModel = ['LRModel']
        self.config.fillNaType=fillNaType.MEAN
        self.config.final_date ='20210105'
        #self.config.runModel=['LRModel','RFModel','NN']

        #self.scaler
        #self.scalerColumnList=[]

    ##資料轉換##
    def dataTransform(self):
        self.dfInputData['key'] =  self.dfInputData['MFG_DATE'].astype(str)+'_' +self.dfInputData['TOOLG_ID'].astype(str) +'_' +self.dfInputData['PROD_ID'].astype(str)
        self.dfInputData['MFG_DATE'] = pd.to_datetime(self.dfInputData['MFG_DATE'],format='%Y%m%d') 
        
        self.dfInputData = self.dfInputData[(self.dfInputData['TOOLG_ID']== 'PK_DUVKrF')& (self.dfInputData['PROD_ID']=='C11MD01A')]

    ##填補遺漏值##
    # def fillnull(self):
    #     self.dfInputData[self.nullColumnlist]=self.dfInputData[self.nullColumnlist].fillna(0)

    ##特徵轉換##
    def featureTransform(self):
        self.dfInputData = pd.get_dummies(self.dfInputData,columns=['TOOLG_ID','PROD_ID'],prefix_sep='_')

    ##準備訓練資料##
    def getTrainingData(self):
        return self.dfInputData[(self.dfInputData['MFG_DATE']<=  pd.to_datetime(self.config.final_date))]

    ##準備測試資料##
    def getTestingData(self):
        return self.dfInputData[(self.dfInputData['MFG_DATE'] >  pd.to_datetime(self.config.final_date))]
if __name__ == "__main__":
    sample=MLSample()
    sample.run()
