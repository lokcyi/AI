from os import replace,path
import pandas as pd
import numpy as np
from BaseClass.MLBase import MLBase ,fillNaType,scalerKind
from Util.EDA import EDA
class MLSample(MLBase,EDA):
    def __init__(self):
        super(MLSample, self).__init__()
        self.log.debug('{}-------------'.format(path.basename(__file__)))
        self.config.reportName = "In line Cycle Time By ToolG"
        self.config.dataSource =  {'DB': 'MPS',
                           'TABLE': 'PPM.dbo.VW_TOOLG_KPI',
                           'CONDITION': [
                                {'column':"MFG_DATE",'operator':">=", 'value': '202001022'},
                                {'column':"INLINE_CT_BY_WAFER",'operator':">", 'value': '0'},#避免抓到NA
                                #  {'column':"MFG_DATE",'operator':"<=", 'value': '202012'},
                                # {'column': 'TOOLG_ID', 'operator': "in", 'value': 'WM_PosCln,WM_PreCln,WM_SW,DI_HDP,DI_HDP_FSG,DI_HDP_HV80,DI_PSG,DI_TD'},
                           ],
        },

        self.config.datafile = "./data/MPS/VW_TOOLG_KPI_CT.csv"
        self.config.targetCol = "INLINE_CT_BY_WAFER"
        self.config.xAxisCol = "MFG_DATE"
        self.config.includeColumns = []
        self.config.excludeColumns = ['TC','INLINE_CT']
        self.config.encoderColumns =['TOOLG_ID'] #vanessa
        self.config.fillNaType = fillNaType.DROPNA
        self.config.scalerKind =scalerKind.MINMAX#scalerKind.MINMAX STANDARD
        self.config.modelFileKey="INLINE_CT_L80AR03A"
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

        # self.config.runModel=['LRModel','NN','XG','CAT','DNN','DNN1k','RFModel']#['LRModel','NN','XG','CAT']# ['DNN','DNN1k','LRModel','NN','RFModel','XG']
        self.config.runModel=['NN','XG','RFModel']

    ##資料轉換##
    def dataTransform(self):
        #異常值清除
        # self.dfInputData['INLINE_CT'] =self.dfInputData['INLINE_CT'].replace(['0', 0], np.nan)
        self.dfInputData['UP_TIME'] =self.dfInputData['UP_TIME'].replace(['0', 0], np.nan)
        self.dfInputData['HOLD_RATE_01'] =  self.dfInputData['HOLD_RATE_01'].replace(np.nan, 0)
        # self.dfInputData['HOLD_RATE_01'] =  self.dfInputData['HOLD_RATE_01'].replace(np.nan, 0)
        # self.dfInputData['HOLD_RATE_HOURLY'] =  self.dfInputData['HOLD_RATE_HOURLY'].replace(np.nan, 0)
        # self.dfInputData['HOLD_RATE'] =  self.dfInputData['HOLD_RATE'].replace(np.nan, 0)

        #目標特徵 刪除空值
        self.dfInputData = self.dfInputData[self.dfInputData[self.config.targetCol].notna()]

        #刪除columns 值是空的()
        self.dfInputData  = self.dfInputData.dropna(axis=1, how='all')

        #資料型態轉型
        self.dfInputData['MFG_DATE'] = self.dfInputData['MFG_DATE'].astype(str)

        # self.__outlier("INLINE_CT")
        # self.__iqrfilter("INLINE_CT",[0,.75])
        # if len(self.dfInputData['UP_TIME'].unique()):
        self.__iqrfilter('UP_TIME',[0,.75])

    def __outlier(self,targetCol):
        mean = self.dfInputData[targetCol].mean()
        sd =self.dfInputData[targetCol].std()
        lower =  mean - 2*sd
        if(lower <0):
            lower=0
        upper = mean + 2*sd
        print( lower,upper)

        self.dfInputData = self.dfInputData[self.dfInputData[targetCol] > lower ]   #[x for x in arr if (x > mean - 2 * sd)]
        self.dfInputData =  self.dfInputData[self.dfInputData[targetCol] < upper] #[x for x in final_list if (x < mean + 2 * sd)]
    def __iqrfilter(self, colname, bounds = [.25, .75]):
        s = self.dfInputData[colname]
        Q1 = self.dfInputData[colname].quantile(bounds[0])
        Q3 = self.dfInputData[colname].quantile(bounds[1])
        IQR = Q3 - Q1
        # print(IQR,Q1,Q3,Q1 - 1.5*IQR,Q3+ 1.5 * IQR)
        if bounds[0]==0:
            self.dfInputData = self.dfInputData[~s.clip(*[Q1,Q3+ 1.5 * IQR]).isin([Q1,Q3+ 1.5 * IQR])]
        else:
            self.dfInputData =  self.dfInputData[~s.clip(*[Q1 - 1.5*IQR,Q3+ 1.5 * IQR]).isin([Q1 - 1.5*IQR,Q3+ 1.5 * IQR])]


   #  ##特徵轉換##
    def featureTransform(self):
         # pass
        self.dfInputDataRaw=  self.dfInputData.copy(deep=False)
        self.dfInputData = pd.get_dummies(self.dfInputData,columns=self.config.encoderColumns,prefix_sep='_') #vanessa

    ##準備訓練資料#### alternative (customize without self.config.TrainCondition)
    def getTrainingData(self):
        pass


    ##準備測試資料## alternative (customize without self.config.TestCondition)
    def getTestingDataRaw(self):
       pass

     ##準備測試資料## alternative (customize without self.config.TestCondition)
    def getTestingData(self):
        pass

# if __name__ == "__main__":
#     sample=MLSample()
#     prodlist =['L15TH02A']
#     toolgList =['DA_AM','DB_Pre','DC_WCVD','DGA_AM_350','DI_HDP','DK_300','DP_SiN','DR_LampA','DS_HDP','DT_BP_G/F','DT_O3','EA_AsherM','EB_Asher','EC_LDD_Logic','EC_LDD_NXP','EC_Via_20','EC_Via_40','EG_LAM_G1','EH_PI','EK_aC','EL_Light','EM_AL_AG','EM_AL_Cln','EM_AL_Depo','EM_W/O','FC_PadOxi','FH_HT','FL_LT','FN_H(F)','FN_SiN(A)','FP_NDPoly','FT_148','IA_MidCur','MA_Al','MA_Al_175','MT_Ti/TiN','MT_TiN','PK_DUVKrF','QW_SEM-PH','SC_M.Jet','WA_PreCln','WB_LiEtch','WDS_160_G2','WH_DSP','WH_EKC','WM_PosCln','WM_PreCln','WN_ContactCln','WN_Co-RMV','WW_NH4OH','XE_Sorter']

#     sample.config.dataSource[0]['CONDITION'] = [
#                 {'column':"MFG_DATE",'operator':">=", 'value': '202001022'},
#                 {'column':"INLINE_CT",'operator':">", 'value': '0'},#避免抓到NA
#                 {'column': 'TOOLG_ID', 'operator': 'in', 'value':  ','.join(['DA_AM','DB_Pre','DC_WCVD','DGA_AM_350','DI_HDP','DK_300','DP_SiN','DR_LampA','DS_HDP','DT_BP_G/F','DT_O3','EA_AsherM','EB_Asher','EC_LDD_Logic','EC_LDD_NXP','EC_Via_20','EC_Via_40','EG_LAM_G1','EH_PI','EK_aC','EL_Light','EM_AL_AG','EM_AL_Cln','EM_AL_Depo','EM_W/O','FC_PadOxi','FH_HT','FL_LT','FN_H(F)','FN_SiN(A)','FP_NDPoly','FT_148','IA_MidCur','MA_Al','MA_Al_175','MT_Ti/TiN','MT_TiN','PK_DUVKrF','QW_SEM-PH','SC_M.Jet','WA_PreCln','WB_LiEtch','WDS_160_G2','WH_DSP','WH_EKC','WM_PosCln','WM_PreCln','WN_ContactCln','WN_Co-RMV','WW_NH4OH','XE_Sorter'])},
#                 {'column': 'PROD_ID', 'operator': '=', 'value': 'L15TH02A'},
#             ]
#     sample.EDAAnalysis()

if __name__ == "__main__":
    sample=MLSample()
    # toolgList = ['CI_ILD_Eba','CI_IMD','CI_STI','CM_WCMP','CN_N2_Cu','CU_Cu','DA_AM','DA_BM','DB_Pre','DC_WCVD','DD_BD_Cu','DD_BLOK_Cu','DGA_AM_350','DGA_AM_400','DI_HDP','DK_300','DP_SiN','DR_LampA','DS_FDY','DT_O3','EA_AsherM','EB_Asher','EC_LDD_Dram','EC_LDD_Logic','EC_Via_40','EG_LAM_G1','EG_LDD','EG_PolyEB_CIS','EG_PolyEB2','EG_STI_DPS','EH_LDD_SP','EH_OxEB','EH_PI','EH_PV','EK_1G','EM_AL_Cln','EM_W/O','EU_Jin_Cu','EU_U_Cu','EU_V_Cu','FC_PadOxi','FH_HT','FL_Cu','FL_LT','FM_ALD','FN_H(F)','FN_SiN(A)','FN_SiN(F)','FP_NDPoly','FQ_PIQ','FT_119','FT_148','IA_MidCur','IBS_HiCur','IBV_HiCur','ID_HiEnrg','MA_Al','MP_Cu','MR_Cu','MS_SIP','MT_Ti/TiN','PB_BARC','PG_UVcure','PH_DUVArF','PH_Immersion','PK_DUVKrF','PT_Marker','PU_I-Line','PW_PIX','QA_ADI','QC_CMP','QC_Cu','QC_ET','QC_TF','QE_ThkMea','QG_Defect','QGK_2365','QP_Defect','QPK_PUMA','QS_SEM','QW_Cu','QW_SEM-ET','QW_SEM-PH','QX_Overly','RAA_RTA','RAM_Anneal','RDA_RTO','RI_BM','RN_RTN_5P/2N','RS_SPA','SC_C/F','SC_M.Jet','WA_PreCln','WB_LiEtch','WDS_160_G2','WE_PreCln','WH_DSP','WH_EKC','WJ_LiEtch','WK_BsEtch','WK_Cu','WL_Resist','WM_PosCln','WM_PreCln','WN_ContactCln','WN_Co-RMV','WQ_CeO2','WT_Cu','WU_W-RMV','WW_HF','WW_NH4OH','WY_DK','XE_Cu','XE_Sorter']#  ,'WM_PreCln','WM_SW','DI_HDP','DI_HDP_FSG','DI_HDP_HV80','DI_PSG','DI_TD']
    #toolgList = ['EC_Via_40','EG_LAM_G1','EM_AL_AG','EM_AL_Cln','EU_U_Cu','FC_PadOxi','FK_LAHO','FM_SiN(A)','FP_NDPoly','IA_MidCur','MA_Al','PK_DUVKrF','SC_M.Jet','WA_PreCln','WH_EKC','WK_Cu','WM_PosCln','WN_Co-RMV','DT_BP_G/F','MT_Ti/TiN','WH_C/F']

    #L15TH02A ==> 'DA_AM','DB_Pre','DC_WCVD','DGA_AM_350','DI_HDP','DK_300','DP_SiN','DR_LampA','DS_HDP','DT_BP_G/F','DT_O3','EA_AsherM','EB_Asher','EC_LDD_Logic','EC_LDD_NXP','EC_Via_20','EC_Via_40','EG_LAM_G1','EH_PI','EK_aC','EL_Light','EM_AL_AG','EM_AL_Cln','EM_AL_Depo','EM_W/O','FC_PadOxi','FH_HT','FL_LT','FN_H(F)','FN_SiN(A)','FP_NDPoly','FT_148','IA_MidCur','MA_Al','MA_Al_175','MT_Ti/TiN','MT_TiN','PK_DUVKrF','QW_SEM-PH','SC_M.Jet','WA_PreCln','WB_LiEtch','WDS_160_G2','WH_DSP','WH_EKC','WM_PosCln','WM_PreCln','WN_ContactCln','WN_Co-RMV','WW_NH4OH','XE_Sorter'
    # toolgList =[ 'DA_AM','DB_Pre','DC_WCVD','DGA_AM_350','DI_HDP','DK_300','DP_SiN','DR_LampA','DS_HDP','DT_BP_G/F','DT_O3','EA_AsherM','EB_Asher','EC_LDD_Logic','EC_LDD_NXP','EC_Via_20','EC_Via_40','EG_LAM_G1','EH_PI','EK_aC','EL_Light','EM_AL_AG','EM_AL_Cln','EM_AL_Depo','EM_W/O','FC_PadOxi','FH_HT','FL_LT','FN_H(F)','FN_SiN(A)','FP_NDPoly','FT_148','IA_MidCur','MA_Al','MA_Al_175','MT_Ti/TiN','MT_TiN','PK_DUVKrF','QW_SEM-PH','SC_M.Jet','WA_PreCln','WB_LiEtch','WDS_160_G2','WH_DSP','WH_EKC','WM_PosCln','WM_PreCln','WN_ContactCln','WN_Co-RMV','WW_NH4OH','XE_Sorter']
    #toolgList =[ 'DT_BP_G/F','DT_O3','EA_AsherM','EB_Asher','EC_LDD_Logic','EC_LDD_NXP','EC_Via_20','EC_Via_40','EG_LAM_G1','EH_PI','EK_aC','EL_Light','EM_AL_AG','EM_AL_Cln','EM_AL_Depo','EM_W/O','FC_PadOxi','FH_HT','FL_LT','FN_H(F)','FN_SiN(A)','FP_NDPoly','FT_148','IA_MidCur',
    # toolgList =[ 'MA_Al','MA_Al_175','MT_Ti/TiN','MT_TiN','PK_DUVKrF','QW_SEM-PH','SC_M.Jet','WA_PreCln','WB_LiEtch','WDS_160_G2','WH_DSP','WH_EKC','WM_PosCln','WM_PreCln','WN_ContactCln','WN_Co-RMV','WW_NH4OH','XE_Sorter']
    # prodlist=['L55GA02A','L15TH02A','D38CL01A']# ,
    # prodlist =['4GDDR3_25D','25D_M81','C11EU16A_WI','C11EU17A_WI','C11EU33A','C11EU39A','C11MD01A','D07AL08A','D07AS04A','D07AS05A','D38AL03A','D38AL05A','D38AS01A','D38AS05A','D38BE23A','D38BR01A','D38CL01A','D38CL02A','D38CL03C','D38CL04A','D38CL05A','D38CL05C','D38CL06A','D38ND01A','D38PA01A','D45AL01A','D45AL03A','D45AL05A','D45AL06A','D45CL01B','D63AL02A','D63BE01A','D63BE03A','D63CL02A','D63CL04A','D63CL05A','D63CL06A','D63PA03A','D63TL02A','F40AL01A','F40MN01A','I14PD02A','I14PD03A','I14PD12A','I14PD14A','I14PD15A','L08AR01A','L09DY46A','L11AR04A','L11GA11A','L11GE16A','L18RU28A','L18RU79A','L18RU80A','L18RU91A','L55GA02A','L80AR03A','L80GA03A','L80GA04A']
    # toolgList=['EC_Via_20']
    toolgList =['EC_Via_20', 'DA_AM','DB_Pre','DC_WCVD','DGA_AM_350','DI_HDP','DK_300','DP_SiN','DR_LampA','DS_HDP','DT_BP_G/F','DT_O3','EA_AsherM','EB_Asher','EC_LDD_Logic','EC_LDD_NXP','EC_Via_20','EC_Via_40','EG_LAM_G1','EH_PI','EK_aC','EL_Light','EM_AL_AG','EM_AL_Cln','EM_AL_Depo','EM_W/O','FC_PadOxi','FH_HT','FL_LT','FN_H(F)','FN_SiN(A)','FP_NDPoly','FT_148','IA_MidCur','MA_Al','MA_Al_175','MT_Ti/TiN','MT_TiN','PK_DUVKrF','QW_SEM-PH','SC_M.Jet','WA_PreCln','WB_LiEtch','WDS_160_G2','WH_DSP','WH_EKC','WM_PosCln','WM_PreCln','WN_ContactCln','WN_Co-RMV','WW_NH4OH','XE_Sorter']
    # prodlist =['L15TH02A']
    for t in toolgList:

        # sample.config.dataSource[0]['CONDITION'] = [
        #     {'column':"MFG_DATE",'operator':">=", 'value': '202001022'},
        #     {'column':"INLINE_CT",'operator':">", 'value': '0'},#避免抓到NA
        #     {'column': 'TOOLG_ID', 'operator': '=', 'value':  t},
        #     {'column': 'PROD_ID', 'operator': '=', 'value':  p},

            #  {'column':"MFG_DATE",'operator':"<=", 'value': '202012'},
            # {'column': 'TOOLG_ID', 'operator': "in", 'value': 'WM_PosCln,WM_PreCln,WM_SW,DI_HDP,DI_HDP_FSG,DI_HDP_HV80,DI_PSG,DI_TD'},
        # ]

        print('condition' ,t )
        sample.config.dataSource[0]['CONDITION'] = [
            {'column':"MFG_DATE",'operator':">=", 'value': '202001022'},
            {'column':"INLINE_CT_BY_WAFER",'operator':">", 'value': '0'},#避免抓到NA
            {'column': 'TOOLG_ID', 'operator': '=', 'value':  t},

        ]
        sample.config.reportName = "Inline Cycle Time BY ToolG({})".format(t)
        sample.config.modelFileKey="INLINE_CT_BY_TOOLG_{}".format(t.replace('/','_'))
        sample.config.InputDataCondition[0]['value'] = t
        sample.run()
        # sample.EDAAnalysis()
        # sample.EDACompare()
    print("***************程式結束***************")

