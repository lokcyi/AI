from os import replace,path
import pandas as pd
import numpy as np
from BaseClass.MLBase import MLBase ,fillNaType,scalerKind
from Util.EDA import EDA
class MLSample(MLBase,EDA):
    def __init__(self):
        super(MLSample, self).__init__()
        self.log.debug('{}-------------'.format(path.basename(__file__)))
        self.config.reportName = "TOOLG MOVE"
        self.config.dataSource =  {'DB': 'MPS',
                           'TABLE': 'PPM.dbo.VW_TOOLG_KPI',
                           'CONDITION': [
                                  {'column':"MFG_DATE",'operator':">=", 'value': '20200122'},
                                  {'column':"MFG_DATE",'operator':"<=", 'value': '20210425'},
                                #   {'column': 'TOOLG_ID', 'operator': "=", 'value': 'PK_DUVKrF'},
                           ],
        },

        self.config.datafile = "./data/MPS/VW_TOOLG_KPI.csv"
        self.config.targetCol = "MOVE_QTY"
        self.config.xAxisCol = "MFG_DATE"
        self.config.includeColumns = []
        # self.config.excludeColumns = ['PROCESS_TIME' ,'WIP_QTY','UP_TIME','TC','INLINE_CT_BY_WAFER','MOVE_QTY_INTERNAL','INLINE_CT','NO_HOLD_WIP', 'BACKUP_BY_RATE','BACKUP_FOR_RATE','REWORK_LOT_RATE','QLIMIT_RATE','SAMPLING_RATE','BATCH_SIZE']
        self.config.excludeColumns = ['INLINE_CT_BY_WAFER','MOVE_QTY_INTERNAL','INLINE_CT']#, 'PROCESS_TIME' , 'TC', 'BACKUP_BY_RATE','BACKUP_FOR_RATE','REWORK_LOT_RATE','QLIMIT_RATE','SAMPLING_RATE','BATCH_SIZE']


        self.config.encoderColumns =['TOOLG_ID'] #vanessa
        # self.config.fillNaType = fillNaType.MEAN
        self.config.fillNaType = fillNaType.DROPNA
        self.config.scalerKind =scalerKind.MINMAX
        self.config.modelFileKey="MPS_MOVE_PK_DUVKrF"
        self.config.forceRetrain = True
        # 初始值篩選條件
        self.config.InputDataCondition= [
            {'column': 'TOOLG_ID', 'operator': "=", 'value': 'PK_DUVKrF'},
        ]
        # 訓練集篩選條件
        self.config.TrainCondition = [
            # {'column':"MFG_DATE",'operator':">=",'value': '201501'},
            {'column':"MFG_DATE",'operator':"<=",'value':'20210411'},
        ]
        # 測試集篩選條件
        self.config.TestCondition = [
            {'column':"MFG_DATE",'operator':">",'value':'20210411'},
        ]

        self.config.runModel=['LRModel','NN','XG','CAT','DNN','DNN1k','RFModel']# ['DNN','DNN1k','LRModel','NN','RFModel','XG']

    ##資料轉換##
    def dataTransform(self):
        # 異常值清除
        self.dfInputData['INLINE_CT'] =self.dfInputData['INLINE_CT'].replace(['0', 0], np.nan)
        self.dfInputData['UP_TIME'] =self.dfInputData['UP_TIME'].replace(['0', 0], np.nan)
        # self.dfInputData = self.__iqrfilter(self.dfInputData,'INLINE_CT',[.25, 1])
        # self.dfInputData = self.__iqrfilter(self.dfInputData,'EQP_UTIL',[.25, 1])
        #刪除Target 是 空值
        self.dfInputData = self.dfInputData[self.dfInputData[self.config.targetCol].notna()]

        #刪除columns 值是空的()
        self.dfInputData  = self.dfInputData.dropna(axis=1, how='all')

        #資料型態轉型
        self.dfInputData['MFG_DATE'] = self.dfInputData['MFG_DATE'].astype(str)

    def __iqrfilter(self,df, colname, bounds = [.25, .75]):
        s = df[colname]
        Q1 = df[colname].quantile(bounds[0])
        Q3 = df[colname].quantile(bounds[1])
        IQR = Q3 - Q1
        # print(IQR,Q1,Q3,Q1 - 1.5*IQR,Q3+ 1.5 * IQR)
        if bounds[0]==0:
            return df[~s.clip(*[Q1,Q3+ 1.5 * IQR]).isin([Q1,Q3+ 1.5 * IQR])]
        else:
            return df[~s.clip(*[Q1 - 1.5*IQR,Q3+ 1.5 * IQR]).isin([Q1 - 1.5*IQR,Q3+ 1.5 * IQR])]

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

   #  ##特徵轉換##
    def featureTransform(self):
         # pass
        self.dfInputDataRaw=  self.dfInputData.copy(deep=False)
        self.dfInputData = pd.get_dummies(self.dfInputData,columns=self.config.encoderColumns,prefix_sep='_') #vanessa

    ##準備訓練資料##
    def getTrainingData(self):
        pass


    ##準備測試資料##
    def getTestingDataRaw(self):
       pass

    def getTestingData(self):
        pass


if __name__ == "__main__":
    sample=MLSample()

    # toolgList = ['PK_DUVKrF','EC_SAC','DA_AM','DI_HDP_HV80','DT_O3','FK_LAHO','WH_EKC']
    # toolgList = ['EC_SAC']#,'EC_SAC','DA_AM','DI_HDP_HV80','DT_O3','FK_LAHO','WH_EKC']
    #toolgList = ['DA_AM','DA_BM','DB_Pre','DGA_AM_350','DI_HDP','DI_HDP_FSG','DI_HDP_HV80','DR_LampA','DS_FDY','DS_HDP','DS_Logic','DT_O3','EB_Asher','EC_SAC',
    # toolgList = ['EC_Via_40','EG_LAM_G1','EM_AL_AG','EM_AL_Cln','EU_U_Cu','FC_PadOxi','FK_LAHO','FM_SiN(A)','FP_NDPoly','IA_MidCur','MA_Al','PK_DUVKrF','SC_M.Jet','WA_PreCln','WH_EKC','WK_Cu','WM_PosCln','WN_Co-RMV','DT_BP_G/F','MT_Ti/TiN','WH_C/F']
    # toolgList = ['DT_BP_G/F','MT_Ti/TiN','WH_C/F']
    toolgList = ['WH_C/F','MT_Ti/TiN']
    # sample.config.runModel=['LRModel']
    for t in toolgList:

        print('condition' ,t )
        sample.config.reportName = "TOOLG_MOVE ({})".format(t)
        sample.config.modelFileKey="TOOLG_MOVE_{}".format(t.replace('/','_'))
        sample.config.InputDataCondition[0]['value'] = t
        sample.run()
        # sample.EDAAnalysis()
        # sample.EDACompare()
    print("***************程式結束***************")
