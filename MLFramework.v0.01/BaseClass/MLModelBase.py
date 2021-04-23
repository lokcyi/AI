import abc
import numpy as np
from pandas import DataFrame
import joblib as joblib
import os
import tensorflow as tf
from sklearn import metrics
from Util.Logger import Logger

class MLModelBase(metaclass=abc.ABCMeta):
    def __init__(self):
        self.log = Logger(name='MLFramework')
        self.log.debug('ML Model Base init..%s' % self.__class__.__name__)

    def doTraining(self,X,y,config):
        self.log.debug('%s doTraining' % self.__class__.__name__)
        modelname=self.__class__.__name__
        h5File="./model/"+modelname+"_{0}.h5".format(config.modelFileKey)
        modelFile="./model/"+modelname+"_{0}.model".format(config.modelFileKey)
        feature_list=config._featureList
        if (os.path.isfile(h5File) and config.forceRetrain==False) :
            print("training "+modelname+":load model from file "+h5File)
            model = tf.keras.models.load_model(h5File)
        else:
            if (os.path.isfile(modelFile) and config.forceRetrain==False) :
                print("training "+modelname+":load model from file "+modelFile)
                model=joblib.load(modelFile)
            else:
                print("training "+modelname+":training model...")
                model = self.training(X, y)
                predicted = model.predict(X)
                r2 = metrics.r2_score(y, predicted)
                self.log.debug('印出模型績效..R2:{0}'.format(r2))


                if hasattr(model,'save'):
                    model.save(h5File)
                else:
                    joblib.dump(model,modelFile)

        ''' # 印出係數
        print(lm.coef_)
        # 印出截距
        print(lm.intercept_ )
        # 模型績效
        mse = np.mean((lm.predict(X) - y) ** 2)
        r_squared = lm.score(X, y)
        adj_r_squared = r_squared - (1 - r_squared) * (X.shape[1] / (X.shape[0] - X.shape[1] - 1))
        # 印出模型績效
        print('MSE:{0}'.format(mse))
        print('R2:{0}'.format(r_squared))
        print('adj_R2:{0}'.format(adj_r_squared))   '''

        if hasattr(model,'feature_importances_'):
            # Get numerical feature importances
            importances = list(model.feature_importances_)
            # List of tuples with variable and importance
            feature_importances = [(feature, round(importance, 3)) for feature, importance in zip(feature_list, importances)]
            # Sort the feature importances by most important first
            feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
            # Print out the feature and importances
            #[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
            df_feature_importances=DataFrame(feature_importances)
            df_feature_importances.columns = ['Variable','重要性_'+modelname]
        else:
            df_feature_importances=DataFrame()

        return model,modelname,df_feature_importances

    @abc.abstractmethod
    def training(self,X,y):
        return NotImplemented