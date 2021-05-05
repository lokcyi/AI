
import abc
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import groupby
import matplotlib.pyplot as plt
import jinja2
import webbrowser
import os
import logging
from enum import Enum

from Util.ModelAnalysis import ModelAnalysis
from Util.EDA import EDA
from Util.Data import Data
from Util.Logger import Logger
from ModelClass import *
import entity.DBEngine as db_engine

class fillNaType(Enum):
    MEAN = 'mean'
    BFILL = 'bfill'
    FFILL = 'ffill'
    DROPNA = 'dropna'
    ZERRO = 'zero'
    MODE = 'mode'
class scalerKind(Enum):
    STANDARD = 'standard'
    MINMAX = 'minmax'
    ROBUST = 'robust'
    MAXABS = 'maxabs'
    NORMAL = 'normal'

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
class MLBase(metaclass=abc.ABCMeta):
    ver="MLFramework v0.01"
    def __init__(self):
        self._config = MLConfig()
        self.log = Logger(name='MLFramework')
        self.log.debug('ML Base init..%s' % self.__class__.__name__)

    @property
    def config(self):
        return self._config
    @config.setter
    def config(self, value):
        self._config = value

    def getDataFromDB(self):
        """
        def 撈取DB
        """
        dataSource=self.config.dataSource[0]
        db_name = dataSource['DB']
        query = ['select * from %s ' % dataSource['TABLE'] ]

        if 'CONDITION' in dataSource.keys():
            if len(dataSource['CONDITION']) > 1:
                for i in range(len(dataSource['CONDITION'])):
                    if i==0:
                        query.append('where ')
                    else:
                        query.append(' AND ')
                    if dataSource['CONDITION'][i]['operator']=='in':
                        query.append('  {} {}  (\'{}\')' .format(dataSource['CONDITION'][i]['column'] ,dataSource['CONDITION'][i]['operator'] ,'\',\''.join(dataSource['CONDITION'][i]['value'].split(','))))
                    else:
                        query.append('  {} {}  \'{}\' '.format(dataSource['CONDITION'][i]['column'] , dataSource['CONDITION'][i]['operator'] , dataSource['CONDITION'][i]['value']))
        conn = db_engine.DBEngine(db_name)
        self.dfInputData = conn.Query(' '.join(query))

        self.dfInputData.to_csv(self.config.datafile, index=False)


    def getMergeDataFile(self):
        self.dfInputData,self.strColumnlist,self.numbericColumnlist,self.nullColumnlist=Data.merge(self.config.dataFiles)
        self.dfInputData.to_csv(self.config.datafile, index=False)

    def getInputData(self):
        self.dfInputData,self.strColumnlist,self.numbericColumnlist,self.nullColumnlist=Data.readData(self.config.datafile)

    def filterData(self):
        self.dfInputData = Data.filterDataframe(self.dfInputData,self.config.InputDataCondition)
        # self.dfInputData,self.strColumnlist,self.numbericColumnlist,self.nullColumnlist=Data.readDataFrame(self.dfInputData)


    @abc.abstractmethod
    def dataTransform(self):
        return NotImplemented

    def filterColumns(self):
        self.dfInputData=Data.filterColumns(self.dfInputData,self.config)
        self.dfInputData, self.strColumnlist, self.numbericColumnlist, self.nullColumnlist = Data.analyzeData(self.dfInputData)
        self.dfTraining = Data.filterColumns(self.dfTraining, self.config)
        self.dfTesting = Data.filterColumns(self.dfTesting, self.config)

    # def scalerData(self):
    #     self.dfInputData=Data.scalerData(self.dfInputData,'MinMaxScaler',self.numbericColumnlist,self.config)
    #     print(self.dfInputData)

    # @abc.abstractmethod
    # def featureTransform(self):
    #     return NotImplemented

    @abc.abstractmethod
    def getTrainingData(self):
        return NotImplemented

    @abc.abstractmethod
    def getTestingData(self):
        return NotImplemented

    def genHTMLReport(self):
        pd.set_option("display.precision", 3)
        htmlRender={}
        for i in range(len(self.config.runModel)):
            mClass=self.config.runModel[i]
            htmlRender['fitable{0}'.format(i + 1)] = (self.mFeatureImportances[mClass].style.render())
            if mClass !='LSTMModel':
                htmlRender['sstable{0}'.format(i+1)]=(ModelAnalysis.sensitivityAnalysis(self.model[mClass],self.mlKind[mClass],self.dfInputData,self.config).style.render())
        htmlRender['ploimage']='{0}_plot.svg'.format(self.config.modelFileKey)


        htmlRender['nowDT']=  datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        htmlRender['reportname']= self.config.reportName
        # Template handling
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=''))
        template = env.get_template('template.html')
        html = template.render(htmlRender)
        #html = template.render(my_table="AAA")
        # Write the HTML file
        path = os.path.abspath('./Report/{0}_report.html'.format(self.config.modelFileKey))
        url = 'file://' + path
        with open(path, 'w') as f:
            f.write(html)
        webbrowser.open(url)

    def __getDATA(self):
        '''
        讀取DB 到local
        '''
        print(bcolors.HEADER + "===" + MLBase.ver + "===================================" + bcolors.ENDC)
        print(bcolors.WARNING + "===[input]讀取DB==================" + bcolors.ENDC)
        self.log.debug("===Data Merge===================%s" % self.__class__.__name__)
        if hasattr(self.config, 'dataSource'):
            self.getDataFromDB()

        '''
        讀取資料合併
        '''
        print(bcolors.HEADER + "===" + MLBase.ver + "===================================" + bcolors.ENDC)
        print(bcolors.WARNING + "===[input]資料合併===================" + bcolors.ENDC)
        self.log.debug("===Data Merge===================%s" % self.__class__.__name__)
        if hasattr(self.config, 'dataFiles'):
            self.getMergeDataFile()

        '''
        讀取資料csv
        '''

        print(bcolors.WARNING + "===[input]讀取資料===================" + bcolors.ENDC)
        self.log.debug("===Fetch Data===================%s" % self.__class__.__name__)
        self.getInputData()
        print(bcolors.WARNING + "===[input]資料過濾===================" + bcolors.ENDC)
        self.log.debug("===Filter Input Data===================%s" % self.__class__.__name__)
        if hasattr(self.config, 'InputDataCondition'):
            self.filterData()

        if self.dfInputData.shape[0] == 0 :
            print(bcolors.WARNING + "資料筆數 : 0" + bcolors.ENDC)
            self.log.debug("資料筆數 : 0   Rows {}, Columns {}".format(self.dfInputData.shape[0],self.dfInputData.shape[1]))
            return
        print(bcolors.WARNING + "===[input]資料轉換===================" + bcolors.ENDC)
        self.log.debug("===Data Transform===================%s" % self.__class__.__name__)
        self.dataTransform()

        '''
        訓練集 & 測試集
        '''

        print(bcolors.WARNING + "===[input]篩選訓練集 & 測試集================" + bcolors.ENDC)
        if hasattr(self.config, 'TrainCondition') & hasattr(self.config, 'TestCondition'):
            self.dfTraining = Data.filterDataframe(self.dfInputData, self.config.TrainCondition)
            self.dfTesting = Data.filterDataframe(self.dfInputData, self.config.TestCondition)
        else:
            self.dfTraining = self.getTrainingData()
            self.dfTesting = self.getTestingData()

        print(bcolors.WARNING + "===[input]過濾資料===================" + bcolors.ENDC)
        self.log.debug("===Data Filter===================%s" % self.__class__.__name__)
        self.filterColumns() # 拆分訓練集 測是集

    def EDAAnalysis(self):
        self.__getDATA()
        # EDA.analysis(df, targetfeat)
        EDA.analysis(self.dfInputData, self.config.targetCol)


    def EDACompare(self):
        self.__getDATA()
        # EDA.analysis(df, targetfeat)
        EDA.compare(self.dfTraining, self.dfTesting, self.config.targetCol)
    '''
    chekck 訓練集 測試集 有資料(如果筆數為0 則停止跑模型)
    '''
    def checkDFSetHasData(self):
        print(bcolors.WARNING +  "資料筆數 : ({},{})".format(self.dfInputData.shape[0],self.dfInputData.shape[1])+ bcolors.ENDC)
        print(bcolors.WARNING + "Training Set  資料筆數 : ({},{})".format(self.dfTraining.shape[0],self.dfTraining.shape[1])+ bcolors.ENDC)
        print(bcolors.WARNING + "Testing Set   資料筆數 : ({},{})".format(self.dfTesting.shape[0],self.dfTesting.shape[1]) +  bcolors.ENDC)
        if self.dfInputData.shape[0] == 0 :
            self.log.debug("Input Set 資料筆數 : 0 ")
            return False
        if self.dfTraining.shape[0] == 0 :

            self.log.debug("Training Set 資料筆數 : 0  ")
            return False
        if self.dfTesting.shape[0] == 0 :

            self.log.debug("Testing Set 資料筆數 : 0 ")
            return False
        return True


    def run(self):
        self.__getDATA()


        '''
        資料預處理
        '''

        print(bcolors.WARNING + "===填補遺漏值==================" + bcolors.ENDC)
        self.log.debug("===填補遺漏值==================%s" % self.__class__.__name__)
        # self.fillnull()
        self.dfTraining = Data.fillnull(self.dfTraining, self.nullColumnlist, self.config.fillNaType.value)
        self.dfTesting = Data.fillnull(self.dfTesting, self.nullColumnlist, self.config.fillNaType.value)
        self.dfOriTesting = self.dfTesting.copy(deep=False)

        if not self.checkDFSetHasData():
            return

        print(bcolors.WARNING + "===特徵縮放===================" + bcolors.ENDC)
        self.log.debug("===特徵縮放===================%s" % self.__class__.__name__)
        # self.scalerData()
        if not hasattr(self.config, 'scalerKind'):
            self.config.scalerKind =scalerKind.MINMAX
        self.dfTraining = Data.scalerData(self.dfTraining, self.config.scalerKind.value,self.numbericColumnlist,self.config, isTrain=True)
        self.dfTesting = Data.scalerData(self.dfTesting, self.config.scalerKind.value,self.numbericColumnlist,self.config, isTrain=False)

        print(bcolors.WARNING + "===特徵轉換===================" + bcolors.ENDC)
        self.log.debug("===特徵轉換===================%s" % self.__class__.__name__)
        # self.featureTransform()
        # self.dfInputDataRaw=  self.dfTraining.copy(deep=False)
        self.dfTraining_eh = Data.featureTransform(self.dfTraining, self.config,True)  # exclude target_cols xAxisCol
        self.dfTraining_eh.to_csv("./log/"+self.config.modelFileKey+'_Training.csv')
        self.dfTesting_eh = Data.featureTransform(self.dfTesting,self.config,False)  # exclude target_cols xAxisCol
        self.dfTesting_eh.to_csv("./log/"+self.config.modelFileKey+'_Testing.csv')
        # self.dfTraining = self.getTrainingData()
        # if hasattr(self.config, 'TrainCondition') and hasattr(self.config, 'TestCondition'):
        #     cols = [ sub['column'] for sub in self.config.TrainCondition+self.config.TestCondition ]
        #     cols = [k for k, g in groupby(sorted(cols))]
        #     self.dfTraining= self.dfTraining.drop(columns=cols)

        self.dfInputData = self.dfTraining_eh
        print(bcolors.WARNING + "===Ready for Training===================" + bcolors.ENDC)
        self.log.debug("===Ready for Training===================%s" % self.__class__.__name__)
        # self.dfTraining_eh= self.dfTraining_eh.drop([x for x in [self.config.xAxisCol] if x in self.dfTraining_eh.columns], axis=1)
        self.X = np.asarray(self.dfTraining_eh)
        self.y = np.asarray(self.dfTraining[self.config.targetCol])


        print(bcolors.WARNING + "===Ready for Testing===================" + bcolors.ENDC)
        self.log.debug("===Ready for Testing===================%s" % self.__class__.__name__)
        # self.dfOriTesting = self.getTestingData()
        # self.dfTesting =  self.dfOriTesting.copy(deep=False)
        # self.dfTesting_eh = self.dfTesting_eh.drop([x for x in [self.config.xAxisCol] if x in self.dfTesting_eh.columns], axis=1)
        self.XTest = np.asarray(self.dfTesting_eh)
        '''
        模型訓練
        '''
        print(bcolors.OKBLUE + "===訓練模型====================" + bcolors.ENDC)
        self.log.debug("===Model Training===================%s" % self.__class__.__name__)
        self.config._featureList=list(self.dfTraining_eh.columns)
        print(bcolors.WARNING + "_featureList : "+ ''.join(self.config._featureList) + bcolors.ENDC)
        self.log.debug("_featureList : {} \n".format( ' , '.join(self.config._featureList)))
        #self.config._featureList=list(self.dfTraining.drop(self.config.targetCol, axis=1).columns)
        self.model={}
        self.mlKind={}
        self.mFeatureImportances={}
        for i in range(len(self.config.runModel)):
            mClass=self.config.runModel[i]
            mObj = getattr(globals()[mClass], mClass)()
            if mClass =='LSTMModel':
                self.X = np.reshape(self.X, (self.X.shape[0], self.X.shape[1], 1))
            self.model[mClass], self.mlKind[mClass], self.mFeatureImportances[mClass] = mObj.doTraining(self.X, self.y, self.config)

        print(bcolors.OKBLUE + "===測試模型====================" + bcolors.ENDC)
        self.log.debug("===Model Testing===================%s" % self.__class__.__name__)

        '''
        模型測試
        '''
        plt.style.use('ggplot')
        plt.figure(figsize=(20,6*len(self.config.runModel)),dpi=60)
        for i in range(len(self.config.runModel)):
            plt.subplot(len(self.config.runModel)*100+10+1+i)
            mClass = self.config.runModel[i]
            if mClass =='LSTMModel':
                self.XTest = np.reshape(self.XTest, (self.XTest.shape[0], self.XTest.shape[1], 1))
            Data.testModel(self.XTest,self.model[mClass],self.mlKind[mClass],self.dfOriTesting,self.config)

        plt.tight_layout()
        plt.savefig('./Report/{0}_plot.svg'.format(self.config.modelFileKey))
        '''
        產生報表
        '''
        print(bcolors.OKBLUE + "===產生報表====================" + bcolors.ENDC)
        self.log.debug("===Create Report===================%s" % self.__class__.__name__)
        self.genHTMLReport()

class MLConfig:
    def __init__(self):
        self._datafile = ""
        self._reportName = "My Report"

    @property
    def datafile(self):
        return self._datafile
    @datafile.setter
    def datafile(self, value):
        self._datafile = value

    @property
    def reportName (self):
        return self._reportName
    @reportName.setter
    def reportName(self, value):
        self._reportName = value

