from os import replace,path
import pandas as pd
import numpy as np
from BaseClass.MLBase import MLBase ,fillNaType,scalerKind

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
        self.config.datafile = "./data/parts20210420/dataset_monthly_20210419_2020.csv"
        self.config.targetCol = "QTY"
        self.config.xAxisCol = "MFG_MONTH"
        self.config.includeColumns = []
        self.config.excludeColumns = ['STOCK_EVENT_TIME','AVG', 'PM', 'TS', 'ENG', 'NST', 'TOOL_ID', 'KEY', 'BACKUP_BY_RATE', 'BACKUP_FOR_RATE', 'RWORK_LOT_RATE', 'SAMPLING_RATE', 'SHIFT1', 'SHIFT2', 'SHIFT3', 'ENG']  #,'CHANGE_RECIPE'
        self.config.encoderColumns =['PART_NO','EQP_NO'] #vanessa
        self.config.fillNaType = fillNaType.MEAN
        self.config.scalerKind =scalerKind.MINMAX
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

        self.config.runModel=['LRModel','NN','XG','CAT']# ['DNN','DNN1k','LRModel','NN','RFModel','XG']
        #self.scaler
        #self.scalerColumnList=[]
        self.dataPreHandler()
    ##資料合併##
    def dataPreHandler(self):
        pass
        # df_parts=pd.read_csv("./data/Parts_EQP_Output_ByMonth_20210407_van.csv")
        # df_parts['MFG_MONTH'] = pd.to_datetime(df_parts['STOCK_EVENT_TIME'].values, format='%Y-%m-%d').astype('period[Q]')
        # df_parts.drop(columns=['STOCK_EVENT_TIME'],inplace=True)
        # df_parts = df_parts.groupby(['PART_NO','EQP_NO','MFG_MONTH']).sum().reset_index()
        # df_EQP=pd.read_csv("./data/ScmTrainingData_Monthly_30_20152021.csv")
        # df_EQP['MFG_MONTH'] = pd.to_datetime(df_EQP['MFG_MONTH'].values, format='%Y%m').astype('period[Q]')
        # df_EQP = df_EQP.groupby(['TOOL_ID','MFG_MONTH']).mean().reset_index()
        # df_merge = pd.merge(df_parts, df_EQP, left_on=['EQP_NO','MFG_MONTH'], right_on=['TOOL_ID','MFG_MONTH'],how="inner")
        # df_merge.to_csv(self.config.datafile, index=False)


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

   #  ##特徵轉換##
   #  def featureTransform(self):
         # pass
   #      self.dfInputDataRaw=  self.dfInputData.copy(deep=False)
   #      self.dfInputData = pd.get_dummies(self.dfInputData,columns=self.config.encoderColumns,prefix_sep='_') #vanessa

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
   sample.config.datafile = "./data/parts20210420/dataset_monthly_20210419.csv"
   # partList = ['85-ECT0010', '85-EKA0190', '85-EKA0200', '85-EKT0100', '42-10-0009', '85-EWA0410', '85-EKT0290', '85-EKA0270', '42-10-0003', '85-EMA0650', '85-EMA0470', '85-EMA0660', '85-EGA0430', '42-10-0014', '85-EWA0220', '85-ECT0060', '85-EWA0960', '85-ECT0750', '85-EWA0380', '85-EWA0440', '85-EKA0610', '85-EKA0620', '85-EKA0530', '85-EKA0600', '85-EKA0630', '85-EWA0390', '85-EWA0430', '85-EWA0400', '85-EMA0430', '85-EKA0260', '85-EKA0010', '85-EKA0020', '85-EKA1020', '85-EMA0440', '42-10-0027', '85-EKT3270', '42-10-0011', '85-EGA0630', '85-EWA1030', '85-EGL0020', '85-EWA1090', '85-ECT2970', '42-10-0108', '85-ECT2980', '85-ECT2990', '85-EMA0310', '85-EGA0930', '85-EMA0410', '85-EMA0420', '85-EMA0320', '85-EWA1100', '85-EGA0410', '85-EWA1080', '85-EGA1070', '85-EGA1050', '85-EGA1060', '85-EGA1120', '85-EKT1020', '85-EKA1280', '85-EVL0010', '85-EGA1040', '85-ECT1290', '85-ECL3110', '85-EKT2070', '85-EKT2060', '85-EKT2040', '85-EKT2090', '85-EKT2050', '42-10-0049', '85-EKA1220', '85-EMA0330', '42-10-0018', '85-EGA0920', '42-10-0065', '85-EMA0250', '85-EAK0750', '85-EHA0100', '85-EWA1170', '85-ECT3150', '85-ENH0200', '85-ECT1420', '85-EKT1050', '85-EKT1040', '85-EMA0170', '85-ECT0860', '85-EGA0290', '85-EKT1060', '85-EKT1030', '85-EWA1240', '85-EVL0050', '85-EVL0070', '85-EGA0540', '85-EAK0740', '85-EAK0190', '85-ECT1180', '85-EVL0090', '85-ECT1200', '85-ECT3700', '85-ECT1190', '85-EVL0100', '85-EVL0110', '85-ECT1450', '85-EGA0230', '85-YYY0180', '85-ECT0770', '85-EGA0530', '72-ES-0860', '85-EAK0850', '85-EKT1410', '85-EWA1110', '85-EMA1020', '85-EMA0240', '85-EMA0780', '85-EHT0020', '85-EMA3620', '85-EWA0460', '85-ECT0820', '85-EWA1390', '85-ECT1500', '70-XX-0240', '42-10-0005', '85-EMA0920',
   # partList = ['85-EMA1350', '85-EGA0640', '85-EGA1100', '85-EGA1370', '85-EWL0080', '85-ECT0830', '85-ECT3230', '85-EWA0881', '85-EMA0200', '85-EGA0270', '85-EWA0920', '85-EMA0220', '85-EGA0490', '85-ECT0960', '85-ECL1920', '85-EMA1340', '70-ZZ-0100', '85-EGA2090', '85-EGA0660', '85-EWA1510', '85-EKA0280', '85-EGT0160', '85-ECT1620', '85-EMA0900', '85-EGL1970', '85-EMA0210', '85-EKA0700', '85-EGT0140', '85-ENH0280', '85-ENH0300', '85-ENH0290', '85-ENH0140', '85-EGT0150', '85-EGH0030', '85-EGH0020', '85-EGH0040', '85-EGH0010', '42-10-0037', '85-EMA3070', '85-EWA1690', '85-EWA1450', '85-EWA0180', '85-EWA0980', '85-EWA0050', '85-EKT2270', '85-EKT3830', '85-ECT0890', '85-EMA0870', '85-EMA0670', '85-EGA0280', '85-ECT0200', '70-XX-0320', '70-XX-1150', '85-EWA0750', '85-EKA0710', '85-EMA0480', '85-EKT0710', '85-EGA1290', '85-ECT1430', '85-EGA1300', '85-EWA0880', '85-EGA0340', '85-EGL0820', '85-ECT3220', '85-EWA1440', '85-EGT0530', '85-EHA0530', '85-ECT2390', '85-EGL0360', '85-EGL0200', '85-EWA1130', '87-RAA4460', '85-EWA1540', '85-EWA1150', '85-EMA0390', '85-EKA0230', '85-EKT0670', '85-EGL1240', '85-EGL0220', '85-EGL0210', '85-EAK0720', '42-10-0069', '70-ZZ-0090', '85-EAK0430', '85-EAK0380', '85-EYS0350', '85-EWA1250', '85-EMA0910', '85-EWA0341', '85-EKT0550', '85-EKT2290', '85-ECT2200', '85-EGA0950', '85-EGA0520', '85-ECT0100', '84-QWH0480', '85-ECL2551', '85-ECL3950', '86-DDA0380', '85-EWL0220', '86-DIA0820', '85-EGL0661', '85-ECT2060', '85-EGA0320', '85-EAK0020', '42-10-0064', '85-EAK0460', '85-EAK0010',
   partList = ['85-EBP0910','85-EWA0340','85-EMA0930','86-DIA0510','85-EGT0560','85-EKT0600','85-EKA0770','85-EHA0790','85-EKA0660','85-EGL0500','85-ECT3020','42-10-0097','42-10-0096','85-EAK0620','70-XX-2070','85-EMA2990','85-EYS0080','85-EMA2780','85-EHT0060','85-EHA0050','85-EKT0530','85-ECT1440','85-ECT1890','85-EGL1590','85-ECT2330','85-ECT2510','85-ECT3190','85-EAK0520','85-EAK1450','42-10-0100','85-EBP0100','42-10-0099','85-EAK0610','85-EWL0461','85-EVL0030','85-EWL0591','85-EWA1541','85-EVL0040','85-EMA1440','85-EVL0020','85-ELK0210','85-EMA0730','85-EMA0850','85-EGL0701','85-ECT1380','85-ECT1310','85-ECT2100','85-ECL3030','85-EBK0060','85-EBP0380','85-ECH0050','85-EAK1430','42-10-0095','85-ECL1311','85-EUT0030','85-EMA2590','85-EWA0890','85-EWA1391','85-YYY0160','85-EMA0981','85-EMA0931','85-EWL0901','85-EMA0860','85-EKT0590','85-EKT0360','85-EMA0230','85-EKT1340','85-EKT0540','85-EKA0080','85-EGT0520','85-ECT2230','85-EGA0260','85-EGL3020','85-EGL2960','85-EGA0580','85-ECT1460','85-ECT2090','85-ECT1560','85-EGL0080','85-ECT0640','85-ECT0110','85-ECL0860','42-10-0098','70-XX-1590','72-EN-0271','85-ECL0811','85-EAK1440','85-ECL2691','85-EAK0490','85-EMA3630','86-DSA2840','85-EWA0490','86-MAA3590','87-RAA1890','85-EKT0300','85-EKT0750','85-EHA1070','85-EKT4280','85-EKT3970','85-EKT2080','85-EKT3900','85-EGL1950','85-ECT1370','85-EGL3530','85-EGL0050','85-EGL1090','85-ECT3380','85-EGL0160','85-EGA0480','85-EAK1090','85-EAK0440','85-EAK0800','85-ECL1240','85-EMA3300','85-ENH0080','85-EWA0461','85-EWL0730','85-EWA0040','85-EWA0181','85-EWL0440','85-EVL0430','86-MAA0910','85-EWA1700','85-EMA3140','85-EMA2330','85-EWA1430','85-EKT0560','85-EKT0470','85-EKT5080','85-EKT1310','85-EHT0040','85-EKA0220','85-EKT2130','85-EKT2400','85-EKT0790','85-EKA1150','85-EGA1341','85-ECT3040','85-ECT2820','85-ECT4630','85-EGL1020','85-ECT2400','85-EGA1570','85-ECT1280','85-ECT1570','85-ECT2520','85-EGL0290','85-EGA1180','85-EGA1210','85-EGA1340','85-EBK0030','85-EBK0350','85-EAK0950','85-EAK0410','85-EAK1300','85-EAK0350','85-EBP0420','73-DS-0180','42-10-0101','85-EBP0390','85-EAK1120','85-EMA1190','86-DIA0810','86-DGA1851','85-EWL0870','86-DSA1490','85-EWA1610','85-EWL0610','85-EVL0510','85-EMA2020','85-EMA1970','87-RIA0080','85-YYY0120','85-EWL0450','85-EMA2660','85-EWA1500','86-DCA2910','85-EWA0580','85-EWL0100','85-EMA2700','85-ELK0010','85-EKT0240','85-EGT0490','85-EHA1080','85-EKT4640','85-EKA1010','85-EMA0050','85-EKT2340','85-EKT1180','85-EKT2380','85-EKA0240','85-EKT1080','85-ELK0090','85-EGL1040','85-EGL2360','85-EGL1880','85-ECT1130','85-EGL0700','85-EGA0350','85-EGL1510','85-ECT2780','85-EGL2040','85-ECT3430','85-ECT3180','85-ECT0920','85-EGA0140','85-EGA0870','85-ECT3270','85-ECT4750','85-ECT1080','85-ECT2440','85-EGL1930','85-ECT1600','85-EGL2210','85-ECT2500','85-EGA1140','85-ECT5290','85-EGA1111','85-ECL2921','85-ECL0650','42-10-0004','73-DE-1140','85-ECL0880','73-MG-0760','85-ECL4100','85-CMA0960','85-ECL0640','85-EAK0050','85-ECL0660','85-EAK0540','85-ECL1340','42-10-0015','85-ECL3320','85-EAK0810','85-ECL5350','85-EAK0880','70-ZZ-0370','85-ECL0630','42-10-0134','87-WFT0780','86-MAA2380','86-DCA0980','85-EWA1160','87-RAA3070','85-EMA3380','85-EMA1760','85-EMA3580','85-EWA0600','85-EWA1460','87-RAA1750','85-EWA1830','87-WFT0250','85-EWL0040','85-YYY0110','85-EMA3740','85-EMA3180','85-EMA3760','86-DGA1850','85-EWA0150','86-DGA3050','85-EML0030','85-EWA0970','85-EWL0590','87-RAA2260','85-EML0050','87-RAA3180','85-EMA3270','85-EYS0120','85-EYS0140','85-EKT1320','85-EKA1350','85-EKT2420','85-EKT2480','85-EMA0270','85-EHA0990','85-EKT1370','85-EHA1060','85-EKA1070','85-EKT0890','85-EKA1380','85-EKA0910','85-EHA0140','85-EKT1120','85-EKT1000','85-EHA1090','85-EGT0060','85-EKT4930','85-EHA0410','85-EKT1150','85-EKA1340','85-EKT2210','85-EKA0970','85-ELK0050','85-EKA0210','85-EGL3611','85-EKT0140','85-EKT1530','85-EKT0960','85-EMA0690','85-EKT1330','85-ELK0230','85-EHA0580','85-ELS0760','85-EHT0030','85-ELS0920','85-EMA0530','85-EHA0080','85-EMA0120','85-EKT1970','85-EKA1200','85-EHA0110','85-ELK0220','85-EKA1190','85-EKT0770','85-EKT0580','85-ECT2290','85-ECT2490','85-ECT2420','85-ECT4710','85-ECT1740','85-ECT4720','85-EGL1600','85-ECT1900','85-ECT4700','85-ECT4730','85-ECT4610','85-ECT4030','85-EGA0600','85-ECT5280','85-ECT2300','85-ECT4330','85-ECT4020','85-ECT5300','85-EGL2270','85-ECT5330','85-EGL3260','85-ECT4340','85-EGA0330','85-ECT2790','85-ECT0900','85-EGA1451','85-ECT4620','85-ECT1980','85-ECT1850','85-EGA1860','85-EGL1290','85-EGA2000','85-EGA0740','85-EGA2080','85-EGL1860','85-EGA2960','85-ECT4640','85-ECT2050','85-EGA0951','85-ECT2830','85-EGA1010','85-ECT4350','85-EGL3000','85-EGA0300','85-EGL3400','85-ECT2080','85-EGL0260','85-EGA1500','84-QWH0970','85-ECL4540','85-ECL3230','85-EBK0070','42-10-0102','85-EBK0380','85-ECL1990','70-ZZ-0440','42-10-0017','85-EBP1020','85-ECL6380','85-ECL0590','85-EAK1490','73-DH-3370','42-10-0146','85-EAK0530','42-10-0016','72-EZ-0870','72-EI-3920','85-EAK0690','85-ECL4230','42-10-0142','85-CMA1010','42-10-0143','85-EAK1000','85-EAK0730','85-ECT0810','42-10-0144','42-10-0145','85-ECL0460','86-DGA4220','86-MBA0770','85-EMA3340','85-EMA3350','87-RAA3310','85-EUT0040','86-DIA1430','85-EMA2060','85-EWA0940','85-EMA3540','87-RAA1891','85-EWL0530','87-REA2160','85-EWA1400','85-EMA1810','85-EMA2160','86-DPA0850','85-EMA1360','85-EMA1930','85-EML0610','85-EWA1730','85-ENH0070','85-ENH0360','89-STK1800','85-EWA1970','85-EWL0950','87-RAA4470','85-EMA1480','85-EMA3250','85-EMA0970','85-EWA0780','85-ENH0500','85-EMA1910','85-EMA2010','86-DIA2910','85-EWA1461','86-DPA0860','85-EMA1270','86-DSA2540','85-EMA1310','86-DTA2130','85-YYY0360','86-MAA1690','86-DAM1090','85-EVL0200','86-DBT2320','86-MBA0810','85-EMA1320','85-EMA1950','85-ENH0270','85-EWA1890','85-EMA0990','85-EUT0020','86-DGA0990','87-RAA4010','85-EWA1620','87-REA1390','85-EMA2960','85-EMA1110','86-DGA2420','85-EMA3730','85-EVL0161','85-EMA1490','85-EWA1750','85-EKT4070','85-EGT0030','85-EGT0250','85-EHA0420','85-EHA0190','85-EGT0120','85-ELS0080','85-EKA0690','85-EGT0290','85-EHA0770','85-EKA0410','85-EKA0771','85-EKT4030','85-EKA0790','85-ELK0040','85-EGL3650','85-ELS0780','85-EKT1820','85-EGT0260','85-EKT1830','85-EGT0310','85-EKT1840','85-EKT0820','85-EKT1850','85-EKA0420','85-EKT1860','85-EMA0840','85-EKT1870','85-EKT4060','85-EKA1080','85-EKT4300','85-EKA1120','85-ELS0010','85-EKT2200','85-EGT0210','85-EGL3660','85-ELS0900','85-EHA1030','85-ELS1080','85-EKT2330','85-EKT0310','85-EKT2390','85-EKT0430','85-EGT0190','85-EGT0010','85-EGT0200','85-EGT0040','85-EKT2590','85-EKA0380','85-EKT3060','85-EKT0990','85-EHA1120','85-EKT1010','85-EKA1400','85-EKA0590','85-EKA1540','85-EHA0330','85-EGT0080','85-EKA1560','85-ECT5530','85-ECT3400','85-EGL1760','85-EGA1281','85-EGL2420','85-ECT2750','85-ECT2020','85-ECT2160','85-EGL1830','85-ECT5600','85-EGA1110','85-ECT3790','85-EGL2941','85-EGA1520','85-EGA0970','85-EGA1540','85-EGL1670','85-ECT2250','85-EGL1810','85-EGA0070','85-ECT2470','85-ECT2810','85-EGL1980','85-EGA2550','85-ECT4550','85-EGA2950','85-EGL2440','85-EGA2990','85-EGL3120','85-ECT2940','85-EGL3480','85-ECT2960','85-ECT2010','85-EGA0650','85-EGL1610','85-EGL0090','85-EGL1700','85-ECT1230','85-EGL1800','85-ECT1610','85-EGL1820','85-ECT4280','85-EGL1850','85-ECT1070','85-ECT4910','85-EGL0380','85-ECT3410','85-EGL0430','85-EGL2070','85-EGA0840','85-EGL2220','85-EGA0160','85-ECT3800','85-ECT3200','85-EGL2421','85-EGL3590','85-EGL2600','85-EGL0790','85-ECT3460','85-ECT1480','85-ECT3880','85-EGA0940','85-EGL3440','85-EGA0240','85-EGA1250','85-EGA0960','85-EGA1280','85-ECL2860','85-ECL1230','85-ECL5330','42-10-0055','85-ECL1550','85-EAK1010','85-ECL4250','85-EAK1040','42-10-0034','85-EAK1190','85-ECL1470','85-EAK1410','85-ECL2071','42-10-0024','42-10-0031','85-EAK0051','85-ECL4401','85-EBK0130','85-ECL5650','85-EBK0180','42-10-0041','85-EAK0070','42-10-0044','85-EBP0350','85-ECL1540','85-EBP0370','85-ECL1690','85-EBP0640','85-ECL2692','85-EBP0660','85-EAK0670','85-EBP0800','85-ECL4150','85-EBP0870','85-ECL4400','85-EBP0890','85-ECL5190','42-10-0138','85-ECL5420','42-10-0139','85-ECL6690','85-ECL0840','42-10-0042','85-EAK0510','85-CMA1160','85-ECL1090','85-ECL1790']
     # 訓練集篩選條件
   sample.config.TrainCondition = [
            {'column':"MFG_MONTH",'operator':">=",'value': '202001'},
            {'column':"MFG_MONTH",'operator':"<=",'value':'202012'},
        ]
        # 測試集篩選條件
   sample.config.TestCondition = [
            {'column':"MFG_MONTH",'operator':">",'value':'202012'},
        ]
   for p in partList:
        sample.config.modelFileKey="Parts_Tools_30_Month_New_{}".format(p)
        sample.config.InputDataCondition[0]['value'] = p
        sample.run()


