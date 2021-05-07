import pandas as pd
import os
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from Util.Logger import Logger
class Featurizer:
    log = Logger(name='MLFramework')
    @staticmethod
    def Select(df,targetfeat,config,limit=0.3):
        cateCols =config.encoderColumns
        '''
        #1. Using Pearson Correlation
        '''
        Featurizer.log.debug("Featurizer=====Using Pearson Correlation")
        # df = pd.concat(( df,  pd.get_dummies(df,columns=cateCols,prefix_sep='_') ), axis=1).drop(cateCols,1)
        df =  pd.get_dummies(df,columns=cateCols,prefix_sep='_')
        cor = df.corr()

        #Correlation with output variable
        cor_target = abs(cor[targetfeat])
        #Selecting highly correlated features
        relevant_features = cor_target[cor_target>limit]
        Featurizer.log.debug(relevant_features.to_string())

        #1. Using Pearson Correlation
        cor = df.corr()

        #Correlation with output variable
        cor_target = abs(cor[targetfeat])
        #Selecting highly correlated features
        relevant_features = cor_target[cor_target>limit]
        Featurizer.log.debug("result============")
        Featurizer.log.debug('\n'+ relevant_features.to_string())

        '''
        Wrapper Method:
        '''
        Featurizer.log.debug("Featurizer=====Using statsmodels OLS model")
        #Adding constant column of ones, mandatory for sm.OLS model
        X = df.drop(targetfeat,1).drop(config.xAxisCol,1)   #Feature Matrix
        y = df[targetfeat]              #Target Variable
        X_1 = sm.add_constant(X)
        #Fitting sm.OLS model

        model = sm.OLS(y,X_1).fit()
        Featurizer.log.debug("result============")
        Featurizer.log.debug(model.pvalues.to_string())


        '''
        # 2.Backward Elimination
        '''
        # cols = list(X.columns)
        # pmax = 1
        # while (len(cols)>0):


        #     p= []
        #     X_1 = X[cols]
        #     X_1 = sm.add_constant(X_1)
        #     print("len(cols) :",len(cols),X_1.shape)
        #     model = sm.OLS(y,X_1).fit()
        #     p = pd.Series(model.pvalues.values[:],index = cols)
        #     pmax = max(p)
        #     feature_with_p_max = p.idxmax()
        #     if(pmax>0.05):
        #         cols.remove(feature_with_p_max)
        #         print("cols.remove(feature_with_p_max):",feature_with_p_max, ' check:',feature_with_p_max in cols)
        #     else:
        #         break
        # selected_features_BE = cols
        # Featurizer.log.debug("result============")
        # Featurizer.log.debug(selected_features_BE)

        '''
        # 2.Embedded Method
        '''
        reg = LassoCV()
        reg.fit(X, y)
        Featurizer.log.debug("Best alpha using built-in LassoCV: %f" % reg.alpha_)
        Featurizer.log.debug("Best score using built-in LassoCV: %f" %reg.score(X,y))
        coef = pd.Series(reg.coef_, index = X.columns)
        Featurizer.log.debug("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
        imp_coef = coef.sort_values(ascending=True)
        Featurizer.log.debug('Lasso coef \n'+imp_coef.to_string())
        # import matplotlib
        # matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
        # imp_coef.plot(kind = "barh")
        # plt.title("Feature importance using Lasso Model")