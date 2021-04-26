import pandas as pd
import numpy as np
from BaseClass.MLBase import MLBase ,fillNaType

class MLSample(MLBase):
    def __init__(self):
        super(MLSample, self).__init__()
        # self.config.dataFiles = {
        #     'files':["./data/Parts_EQP_Output_ByMonth_20210407_van.csv"
        #              ,"./data/ScmTrainingData_Monthly_30_20152021.csv"
        #             ],
        #     'relations':[
        #             [['MFG_MONTH','EQP_NO'],['MFG_MONTH','TOOL_ID']]
        #             ,[['MFG_MONTH'],['MFG_MONTH']]
        #             ]
        # }
        '''
        初始值篩選條件
        '''
        self.config.InputDataCondition= [
            {'column': 'PART_NO', 'operator': "=", 'value': '85-EKA0270'},
        ]

        self.config.datafile = "./data/Parts_EQP_Output_ByMonth_20210407_van.csv"
        self.config.targetCol = 'QTY'
        self.config.xAxisCol = "MFG_MONTH"
        self.config.includeColumns = []
        self.config.excludeColumns =['PM','TS','ENG','NST']
        # self.config.fillNaType=fillNaType.MEAN  ##填補遺漏值##
        self.config.modelFileKey="Parts_LSTM"

        self.config.forceRetrain=True

        self.config.runModel = ['LSTMModel']  #['DNN','DNN1k','LRModel','NN','RFModel','XG']
        self.config.LSTM_look_back = 2
        #self.scaler
        #self.scalerColumnList=[]

    ##資料轉換##
    def dataTransform(self):
        periodType = 'period[M]'
        df = self.dfInputData
        df['MFG_MONTH'] = pd.to_datetime(df['MFG_MONTH'].values, format='%Y%m').astype(periodType)
        # df=df[df['PART_NO']==PARTNO]
        df =df.groupby(['MFG_MONTH']).sum()
        df = df.sort_values(by=['MFG_MONTH'], ascending=[True])
        df.reindex(pd.period_range(df.index[0],df.index[-1],freq='M'))
        df = df.reset_index()
        # df.drop(columns=['PM','TS','ENG','NST'],inplace=True)
        self.dfInputData = df

    ##特徵轉換##
    def featureTransform(self):
        self.dfInputDataRaw=  self.dfInputData.copy(deep=False)
        # self.dfInputData = pd.get_dummies(self.dfInputData,columns=['EQP_NO','PART_NO'],prefix_sep='_')

    ##準備訓練資料##
    def getTrainingData(self):

        train  = self.dfInputDataRaw[self.dfInputDataRaw['MFG_MONTH']<'202101']
        training_set = train.iloc[:, 1:2].values  # 72
         #Featuring Scaling(LSTM 是對 target 做 scaling)
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler(feature_range = (0,1))
        scaled_training_data = sc.fit_transform(training_set)
        #Creating Data Structure with 60 Time Stamps and 1 output
        x_train = []
        y_train = []
        #
        for i in range(self.config.LSTM_look_back,train.shape[0]):
            x_train.append(scaled_training_data[i-self.config.LSTM_look_back:i, 0])
            y_train.append(scaled_training_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)

        #Reshaping
        # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # (74,1)==>  筆數 , (look_back,1) (74, 1, 1)
        # Append suffix / prefix to strings in list

        # pre_res = ['X' + sub.str() for sub in range(1, x_train.shape[1]+1)]
        pre_res = ['X'+str(i+1) for i in range(x_train.shape[1])]
        df_train = pd.DataFrame(x_train, columns = pre_res)
        df_train[self.config.targetCol] = y_train

        return df_train

    ##準備測試資料##
    def getTestingDataRaw(self):
        train  = self.dfInputDataRaw[self.dfInputDataRaw['MFG_MONTH']<'202101']
        test  = self.dfInputDataRaw[self.dfInputDataRaw['MFG_MONTH']>='202101']
        real_test_TargetValue = test[self.config.targetCol].iloc[:].values

        dataset_total = pd.concat((train[self.config.targetCol], test[self.config.targetCol]), axis = 0)
        inputs = dataset_total[len(dataset_total) - len(test) - self.config.LSTM_look_back:].values
        inputs = inputs.reshape(-1,1)
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler(feature_range = (0,1))
        training_set = train.iloc[:, 1:2].values  # 72
        sc.fit_transform(training_set)
        inputs = sc.transform(inputs)
        X_test = []
        for i in range(self.config.LSTM_look_back, inputs.shape[0]):
            X_test.append(inputs[i-self.config.LSTM_look_back:i, 0])
        X_test = np.array(X_test)
        # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        pre_res = ['X'+str(i+1) for i in range(X_test.shape[1])]

        df_test = pd.DataFrame(X_test, columns = pre_res)

        df_test[self.config.targetCol]=real_test_TargetValue
        df_test[self.config.xAxisCol] = test[self.config.xAxisCol].dt.strftime('%Y%m').values
        return df_test

    def getTestingData(self):
        return self.getTestingDataRaw()

if __name__ == "__main__":
    sample=MLSample()
    sample.run()
