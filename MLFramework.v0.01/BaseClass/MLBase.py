
import abc
import pandas as pd
import numpy as np
from itertools import groupby
import matplotlib.pyplot as plt
import jinja2
import webbrowser
import os
import logging
from enum import Enum
from Util.ModelAnalysis import ModelAnalysis
from Util.Data import Data
from Util.Logger import Logger
from ModelClass import *
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

    def getMergeDataFile(self):
        self.dfInputData,self.strColumnlist,self.numbericColumnlist,self.nullColumnlist=Data.merge(self.config.dataFiles)
        self.dfInputData.to_csv(self.config.datafile, index=False)

    def getInputData(self):
        self.dfInputData,self.strColumnlist,self.numbericColumnlist,self.nullColumnlist=Data.readData(self.config.datafile)

    def filterData(self):
        self.dfInputData = Data.filterDataframe(self.dfInputData,self.config.InputDataCondition)
        self.dfInputData,self.strColumnlist,self.numbericColumnlist,self.nullColumnlist=Data.readDataFrame(self.dfInputData)

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

    def run(self):
        print(bcolors.HEADER + "===" + MLBase.ver + "===================================" + bcolors.ENDC)
        print(bcolors.WARNING + "===[input]資料合併===================" + bcolors.ENDC)
        self.log.debug("===Data Merge===================%s" % self.__class__.__name__)
        if hasattr(self.config, 'dataFiles'):
            self.getMergeDataFile()
        print(bcolors.WARNING + "===[input]讀取資料===================" + bcolors.ENDC)
        self.log.debug("===Fetch Data===================%s" % self.__class__.__name__)
        self.getInputData()
        print(bcolors.WARNING + "===[input]資料過濾===================" + bcolors.ENDC)
        self.log.debug("===Filter Input Data===================%s" % self.__class__.__name__)
        if hasattr(self.config, 'InputDataCondition'):
            self.filterData()
        print(bcolors.WARNING + "===[input]資料轉換===================" + bcolors.ENDC)
        self.log.debug("===Data Transform===================%s" % self.__class__.__name__)
        self.dataTransform()


        print(bcolors.WARNING + "===[input]篩選訓練集 & 測試集================" + bcolors.ENDC)
        self.dfTraining = Data.filterDataframe(self.dfInputData, self.config.TrainCondition)
        self.dfTesting = Data.filterDataframe(self.dfInputData, self.config.TrainCondition)

        print(bcolors.WARNING + "===[input]過濾資料===================" + bcolors.ENDC)
        self.log.debug("===Data Filter===================%s" % self.__class__.__name__)
        self.filterColumns()

        '''
        資料預處理
        '''

        print(bcolors.WARNING + "===填補遺漏值==================" + bcolors.ENDC)
        self.log.debug("===填補遺漏值==================%s" % self.__class__.__name__)
        # self.fillnull()
        self.dfTraining = Data.fillnull(self.dfTraining, self.nullColumnlist, self.config.fillNaType.value)
        self.dfTesting = Data.fillnull(self.dfTesting, self.nullColumnlist, self.config.fillNaType.value)
        self.dfOriTesting = self.dfTesting.copy(deep=False)

        print(bcolors.WARNING + "===特徵縮放===================" + bcolors.ENDC)
        self.log.debug("===特徵縮放===================%s" % self.__class__.__name__)
        # self.scalerData()
        self.dfTraining = Data.scalerData(self.dfTraining, self.numbericColumnlist,self.config, isTrain=True)
        self.dfTesting = Data.scalerData(self.dfTesting, self.numbericColumnlist,self.config, isTrain=False)

        print(bcolors.WARNING + "===特徵轉換===================" + bcolors.ENDC)
        self.log.debug("===特徵轉換===================%s" % self.__class__.__name__)
        # self.featureTransform()
        # self.dfInputDataRaw=  self.dfTraining.copy(deep=False)
        self.dfTraining_eh = Data.featureTransform(self.dfTraining, self.config,True)
        self.dfTesting_eh = Data.featureTransform(self.dfTesting,self.config,False)

        # self.dfTraining = self.getTrainingData()
        # if hasattr(self.config, 'TrainCondition') and hasattr(self.config, 'TestCondition'):
        #     cols = [ sub['column'] for sub in self.config.TrainCondition+self.config.TestCondition ]
        #     cols = [k for k, g in groupby(sorted(cols))]
        #     self.dfTraining= self.dfTraining.drop(columns=cols)
        print(bcolors.WARNING + "===Ready for Training===================" + bcolors.ENDC)
        self.log.debug("===Ready for Training===================%s" % self.__class__.__name__)
        self.dfTraining_eh= self.dfTraining_eh.drop([x for x in [self.config.xAxisCol] if x in self.dfTraining_eh.columns], axis=1)
        self.X = np.asarray(self.dfTraining_eh).astype('float32')
        self.y = np.asarray(self.dfTraining[self.config.targetCol])


        print(bcolors.WARNING + "===Ready for Testing===================" + bcolors.ENDC)
        self.log.debug("===Ready for Testing===================%s" % self.__class__.__name__)
        # self.dfOriTesting = self.getTestingData()
        # self.dfTesting =  self.dfOriTesting.copy(deep=False)
        self.dfTesting_eh = self.dfTesting_eh.drop([x for x in [self.config.xAxisCol] if x in self.dfTesting_eh.columns], axis=1)
        self.XTest = np.asarray(self.dfTesting_eh).astype('float32')

        print(bcolors.OKBLUE + "===訓練模型====================" + bcolors.ENDC)
        self.log.debug("===Model Training===================%s" % self.__class__.__name__)
        # self.config._featureList=list(self.dfTraining_eh.drop(self.config.targetCol, axis=1).columns)
        self.config._featureList=list(self.dfTraining.drop(self.config.targetCol, axis=1).columns)
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
        print(bcolors.OKBLUE + "===產生報表====================" + bcolors.ENDC)
        self.log.debug("===Create Report===================%s" % self.__class__.__name__)
        self.genHTMLReport()

class MLConfig:
    def __init__(self):
        self._datafile = ""

    @property
    def datafile(self):
        return self._datafile
    @datafile.setter
    def datafile(self, value):
        self._datafile = value



