from os import replace,path
import pandas as pd
import numpy as np
from BaseClass.MLBase import MLBase ,fillNaType,scalerKind

class MLSample(MLBase):
    def __init__(self):
        super(MLSample, self).__init__()
        self.log.debug('{}-------------'.format(path.basename(__file__)))
        self.config.dataSource =  {'DB': 'MPS',
                           'TABLE': 'PPM.dbo.VW_PROD_KPI',
                           'CONDITION': [
                                  {'column':"MFG_DATE",'operator':">=", 'value': '20200101'},
                                 #  {'column':"MFG_DATE",'operator':"<=", 'value': '202012'},
                                  {'column': 'TOOLG_ID', 'operator': "=", 'value': 'PK_DUVKrF'},
                           ],
        },

        self.config.datafile = "./data/MPS/VW_PROD_KPI.csv"
        self.config.targetCol = "INLINE_CT"
        self.config.xAxisCol = "MFG_DATE"
        self.config.includeColumns = []
        # self.config.excludeColumns = ['STOCK_EVENT_TIME','AVG', 'PM', 'TS', 'ENG', 'NST', 'TOOL_ID', 'KEY', 'BACKUP_BY_RATE', 'BACKUP_FOR_RATE', 'RWORK_LOT_RATE', 'SAMPLING_RATE', 'SHIFT1', 'SHIFT2', 'SHIFT3', 'ENG']  #,'CHANGE_RECIPE'
        self.config.excludeColumns = ['TC']
        self.config.encoderColumns =['PART_NO','EQP_NO'] #vanessa
        self.config.fillNaType = fillNaType.MEAN
        self.config.scalerKind =scalerKind.MINMAX
        self.config.modelFileKey="INLINE_CT_L80AR03A"
        self.config.forceRetrain = True
        # 初始值篩選條件
        self.config.InputDataCondition= [
            {'column': 'PROD_ID', 'operator': "=", 'value': 'L80AR03A'},
        ]
        # 訓練集篩選條件
        self.config.TrainCondition = [
            # {'column':"MFG_DATE",'operator':">=",'value': '201501'},
            {'column':"MFG_DATE",'operator':"<=",'value':'20210115'},
        ]
        # 測試集篩選條件
        self.config.TestCondition = [
            {'column':"MFG_DATE",'operator':">",'value':'20210115'},
        ]

        self.config.runModel=['LRModel','NN','XG','CAT']# ['DNN','DNN1k','LRModel','NN','RFModel','XG']

    ##資料轉換##
    def dataTransform(self):
        #異常值清除
        self.dfInputData['INLINE_CT'] =self.dfInputData['INLINE_CT'].replace(['0', 0], np.nan)
        #目標特徵 刪除空值
        self.dfInputData = self.dfInputData[self.dfInputData[self.config.targetCol].notna()]

        #刪除columns 值是空的()
        self.dfInputData  = self.dfInputData.dropna(axis=1, how='all')

        #資料型態轉型
        self.dfInputData['MFG_DATE'] = self.dfInputData['MFG_DATE'].astype(str)
    def __outlier(self):
        mean = self.dfInputData[self.config.targetCol].mean()
        sd =self.dfInputData[self.config.targetCol].std()
        lower =  mean - 2*sd
        if(lower <0):
            lower=0
        upper = mean + 2*sd
        print( lower,upper)

        self.dfInputData = self.dfInputData[self.dfInputData[self.config.targetCol] > lower ]   #[x for x in arr if (x > mean - 2 * sd)]
        self.dfInputData =  self.dfInputData[self.dfInputData[self.config.targetCol] < upper] #[x for x in final_list if (x < mean + 2 * sd)]
    def __iqrfilter(df, colname, bounds = [.25, .75]):
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
        # df = self.dfInputData.copy(deep=False)
        # for c in self.config.TrainCondition  :
        #     if c['operator'] == "=":
        #        df =df[df[c['column']] == c['value']]
        #     elif c['operator']  == "=!":
        #        df =df[df[c['column']] != c['value']]
        #     elif c['operator']  == "<=":
        #        df =df[df[c['column']] <= c['value']]
        #     elif c['operator']  == "<":
        #        df =df[df[c['column']] < c['value']]
        #     elif  c['operator']  == ">=":
        #        df =df[df[c['column']] >=c['value']]
        #     elif c['operator']  ==  ">":
        #        df = df[df[c['column']] > c['value']]

        # df.to_csv('./log/trainingDATA_{}.csv'.format(self.config.modelFileKey))
        # return  df

    ##準備測試資料##
    def getTestingDataRaw(self):
       pass
   #       #super(MLSample, self).getTestingDataRaw() #calling method of parent class
   #      df = self.dfInputDataRaw.copy(deep=False)
   #      for c in self.config.TestCondition  :
   #          if c['operator'] == "=":
   #             df =df[df[c['column']] == c['value']]
   #          elif c['operator']  == "=!":
   #             df =df[df[c['column']] != c['value']]
   #          elif c['operator']  == "<=":
   #             df =df[df[c['column']] <= c['value']]
   #          elif c['operator']  == "<":
   #             df =df[df[c['column']] < c['value']]
   #          elif  c['operator']  == ">=":
   #             df =df[df[c['column']] >=c['value']]
   #          elif c['operator']  ==  ">":
   #             df = df[df[c['column']] > c['value']]
   #      return df

    def getTestingData(self):
        pass
   #      df = self.dfInputData.copy(deep=False)
   #      for c in self.config.TestCondition  :
   #          if c['operator'] == "=":
   #             df =df[df[c['column']] == c['value']]
   #          elif c['operator']  == "=!":
   #             df =df[df[c['column']] != c['value']]
   #          elif c['operator']  == "<=":
   #             df =df[df[c['column']] <= c['value']]
   #          elif c['operator']  == "<":
   #             df =df[df[c['column']] < c['value']]
   #          elif  c['operator']  == ">=":
   #             df =df[df[c['column']] >=c['value']]
   #          elif c['operator']  ==  ">":
   #             df = df[df[c['column']] > c['value']]
   #      df.to_csv('./log/testingDATA_{}.csv'.format(self.config.modelFileKey))
   #      return  df


if __name__ == "__main__":
   sample=MLSample()


   prodIdList=['L80AR03A']
     # 訓練集篩選條件
#    sample.config.TrainCondition = [
#             {'column':"MFG_MONTH",'operator':">=",'value': '202001'},
#             {'column':"MFG_MONTH",'operator':"<=",'value':'202012'},
#         ]
#         # 測試集篩選條件
#    sample.config.TestCondition = [
#             {'column':"MFG_MONTH",'operator':">",'value':'202012'},
#         ]
   for p in prodIdList:
        sample.config.modelFileKey="INLINE_CT_{}".format(p)
        sample.config.InputDataCondition[0]['value'] = p
        sample.run()
   print("***************程式結束***************")

