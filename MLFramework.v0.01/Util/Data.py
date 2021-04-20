import pandas as pd
import os
import numpy as np
import joblib as joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler #平均&變異數標準化 平均值為0，方差為1。
from sklearn.preprocessing import MinMaxScaler #最小最大值標準化[0,1]
from sklearn.preprocessing import RobustScaler #中位數和四分位數標準化
from sklearn.preprocessing import MaxAbsScaler #絕對值最大標準化
from Util.Logger import Logger

class Data:
    log = Logger(name='MLFramework')
    @staticmethod
    def readData(inputfile):
        Data.log.debug('readData %s' % inputfile)
        df = pd.read_csv(inputfile)
        df=df.dropna(axis=1,how='all')
        df.info()
        return Data.analyzeData(df)
    @staticmethod
    def readDataFrame(df):
        df=df.dropna(axis=1,how='all')
        return Data.analyzeData(df)
    @staticmethod
    def merge(dataFiles):
        index =0
        for dfFile in dataFiles['files']:
            print(dfFile)
            if index ==0:
                _dfInputData1,_strColumnlist1,_numbericColumnlist1,_nullColumnlist1=Data.readData(dfFile)

                # _df_result.
            else:
                datasetRels = dataFiles['relations'][index-1]
                _dfInputData2,_strColumnlist2,_numbericColumnlist2,_nullColumnlist2=Data.readData(dfFile)
                _dfInputData1.set_index(datasetRels[0])
                _dfInputData2.set_index(datasetRels[1])
                df_merge = Data.mergeDataFrame(_dfInputData1,_dfInputData2,datasetRels[0],datasetRels[1])
                _dfInputData1 = df_merge.copy(deep=False)
            index+=1

        return  Data.analyzeData(df_merge)
    @staticmethod
    def mergeDataFrame(dfleft,dfright,LeftKeys,RightKeys):
        # dfright.columns = [str(col) + '_'+joinTableName for col in df.columns]
        df_merge = pd.merge(dfleft, dfright, left_on=LeftKeys, right_on=RightKeys,how="inner")
        return df_merge
    @staticmethod
    def analyzeData(df):
        print('非數值欄位：')
        strColumnlist=df.select_dtypes(exclude=['int64','float64']).columns.tolist()
        print(strColumnlist)
        print('數值欄位：')
        numbericColumnlist=df.select_dtypes(include=['int64','float64']).columns.tolist()
        print(numbericColumnlist)
        print('包含ＮＵＬＬ的欄位：')
        nullColumnlist=df.columns[df.isna().any()].tolist()
        print(nullColumnlist)
        print('===================================================')
        return df, strColumnlist, numbericColumnlist, nullColumnlist
    @staticmethod
    def filterDataframe(df,condition):
        for c in condition :
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
               df =df[df[c['column']] >c['value']]
        return df
    @staticmethod
    def filterColumns(df,config):
        includeColumns=config.includeColumns
        excludeColumns=config.excludeColumns
        if (len(includeColumns)>0):
            df=df[includeColumns]
        df=df.drop(columns=excludeColumns)
        return df
    @staticmethod
    def scalerData(df,scalerKind,numbericColumnlist,config):
        target_cols=config.targetCol
        scalerColumnlist = [ele for ele in numbericColumnlist if ele not in target_cols]
        scaler = MinMaxScaler()
        scaler.fit(df[scalerColumnlist])
        df[scalerColumnlist]= scaler.transform(df[scalerColumnlist])
        return df
    @staticmethod
    def accsum(def_result,target_cols):
        _accsum=0
        for index,row in def_result.iterrows():
            #避免當分母為0 會無法計算
            if row[target_cols]==0 and row['Predict']==0 :
                row[target_cols] =1
                row['Predict'] =1
            elif row[target_cols] ==0 and row['Predict']!=0:
                row[target_cols]  =0.00001

            if row[target_cols] <0 :
                row[target_cols]  =0.00001

            if row['Predict'] <0 :
                row['Predict']  =0

            if 1- abs((row['Predict'] - row[target_cols])/row[target_cols] ) >0 :
                _accsum+=(1- abs((row['Predict'] - row[target_cols])/row[target_cols] ))

        return round(_accsum*100/def_result.shape[0],2)

    @staticmethod
    def testModel(XTest,model,mlKind,df,config):
        yTest=model.predict(XTest)
        df2=df.copy(deep=False)
        df2.insert(len(df2.columns), 'Predict', yTest)
        plt.title((mlKind+":{0}%").format(Data.accsum(df2,config.targetCol)))
        plt.xlabel(config.xAxisCol)
        plt.xticks(rotation=90)
        plt.ylabel(config.targetCol)
        t = df2[config.xAxisCol].to_numpy()+'_'+np.arange(len(XTest)).astype(str)  # 创建t变量
        plt.plot(t,df2['Predict'], label = mlKind, color='red', marker='.',linewidth = '0.5')
        plt.plot(t,df2[config.targetCol], label = "ACT", color='blue', marker='x',linewidth = '0')
        plt.legend()
        plt.ylim(bottom=0)
        df2.to_csv('./Report/'+config.modelFileKey+'_'+mlKind+'.csv',index=False)
        print(mlKind+" Test acc%:",mlKind,Data.accsum(df2,config.targetCol))
        _acc = mlKind,Data.accsum(df2,config.targetCol)
        Data.log.debug(mlKind+" Test acc: :%.2f" % _acc[1])

        _accsum=0
        def_result_summary = df2.groupby(config.xAxisCol, as_index=False).sum().reset_index()[[config.xAxisCol,config.targetCol,'Predict']]
        if(def_result_summary.shape[0]>1):
            _acc = mlKind,Data.accsum(def_result_summary,config.targetCol)
            Data.log.debug(mlKind+" Test group by x-axis acc: :%.2f" % _acc[1])
            print(mlKind+" Test group by x-axis acc: :%.2f" % _acc[1])

        def_result_summary = df2[[config.targetCol,'Predict']].sum()
        if(def_result_summary[config.targetCol]!=0):
            totol_acc = (1- abs(def_result_summary['Predict'] -def_result_summary[config.targetCol])/def_result_summary[config.targetCol])*100
            print(mlKind+" Test Aggreation acc :%f ",totol_acc)
            Data.log.debug(mlKind+" Test Aggreation acc :%f " % totol_acc)
            dfraw=pd.read_csv(config.datafile)



            # partsData = dfraw[dfraw['PART_NO']==config.partno]
            # qtymean = partsData[config.targetCol].mean()


            # totol_acc = (1- abs(qtymean -def_result_summary['QTY'])/def_result_summary['QTY'])*100
            # print(mlKind+" Test Mean (%f) acc :%f  ", (qtymean ,totol_acc))
            # Data.log.debug(mlKind+" Test Mean (%f)  acc :%f " % (qtymean ,totol_acc))