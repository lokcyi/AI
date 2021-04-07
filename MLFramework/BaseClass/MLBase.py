
import abc
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import jinja2
import webbrowser
import os  
from Util.ModelAnalysis import ModelAnalysis
from Util.Data import Data  
from ModelClass import *
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
    @property
    def config(self):
        return self._config
    @config.setter
    def config(self, value):
        self._config = value   
   
    def getInputData(self):
        self.dfInputData,self.strColumnlist,self.numbericColumnlist,self.nullColumnlist=Data.readData(self.config.datafile) 

    @abc.abstractmethod
    def dataTransform(self):
        return NotImplemented    

    def filterData(self):    
        self.dfInputData=Data.filterColumns(self.dfInputData,self.config)
        self.dfInputData,self.strColumnlist,self.numbericColumnlist,self.nullColumnlist=Data.analyzeData(self.dfInputData)  
    
    @abc.abstractmethod
    def fillnull(self):
        return NotImplemented    

    def scalerData(self):
        self.dfInputData=Data.scalerData(self.dfInputData,'MinMaxScaler',self.numbericColumnlist,self.config)
        print(self.dfInputData)

    @abc.abstractmethod
    def featureTransform(self):
        return NotImplemented 

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
            htmlRender['fitable{0}'.format(i+1)]=(self.mFeatureImportances[mClass].style.render())
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
        print(bcolors.HEADER + "==="+MLBase.ver+"===================================" + bcolors.ENDC)
        print(bcolors.WARNING + "===讀取資料===================" + bcolors.ENDC)
        self.getInputData()
        print(bcolors.WARNING + "===資料轉換===================" + bcolors.ENDC)
        self.dataTransform()
        print(bcolors.WARNING + "===過濾資料===================" + bcolors.ENDC)
        self.filterData()
        print(bcolors.WARNING + "===填補遺漏值==================" + bcolors.ENDC)
        self.fillnull()
        self.dfInputData.info()
        print(bcolors.WARNING + "===特徵縮放===================" + bcolors.ENDC)
        self.scalerData()
        print(bcolors.WARNING + "===特徵轉換===================" + bcolors.ENDC)
        self.featureTransform()
        self.dfInputData.info()
        print(bcolors.WARNING + "===準備訓練資料================" + bcolors.ENDC)
        self.dfTraining=self.getTrainingData()
        self.dfTraining=self.dfTraining.drop(columns=[self.config.xAxisCol])     
        self.X = np.asarray(self.dfTraining.drop(self.config.targetCol, axis=1))
        self.y = np.asarray(self.dfTraining[self.config.targetCol])
        print(bcolors.WARNING + "===準備測試資料================" + bcolors.ENDC)
        self.dfTesting=self.getTestingData()
        self.dfOriTesting=self.dfTesting          
        self.dfTesting=self.dfTesting.drop(columns=[self.config.xAxisCol])
        self.XTest = np.asarray(self.dfTesting.drop(self.config.targetCol, axis=1))
        print(bcolors.OKBLUE + "===訓練模型====================" + bcolors.ENDC)
        self.config._featureList=list(self.dfTraining.drop(self.config.targetCol, axis=1).columns)       
        self.model={}
        self.mlKind={}
        self.mFeatureImportances={}
        for i in range(len(self.config.runModel)):
            mClass=self.config.runModel[i]
            mObj=getattr(globals()[mClass],mClass)()
            self.model[mClass],self.mlKind[mClass],self.mFeatureImportances[mClass]=mObj.doTraining(self.X,self.y,self.config) 

        print(bcolors.OKBLUE + "===測試模型====================" + bcolors.ENDC)
        plt.style.use('ggplot')
        plt.figure(figsize=(8,6),dpi=120)
        for i in range(len(self.config.runModel)):            
            plt.subplot(331+i)
            mClass=self.config.runModel[i]
            Data.testModel(self.XTest,self.model[mClass],self.mlKind[mClass],self.dfOriTesting,self.config) 
         
        plt.tight_layout()
        plt.savefig('./Report/{0}_plot.svg'.format(self.config.modelFileKey))  
        print(bcolors.OKBLUE + "===產生報表====================" + bcolors.ENDC)
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



 