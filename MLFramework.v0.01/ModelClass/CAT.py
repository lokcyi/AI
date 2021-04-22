from BaseClass.MLModelBase import MLModelBase
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor
from Util.Logger import Logger
from sklearn.metrics import r2_score
class CAT(MLModelBase):
    def __init__(self):
        MLModelBase.__init__(self)
    def training(self,X,y):
        X_train, X_test,y_train,y_test = train_test_split(X,y,test_size =0.2,random_state=587)

        # cv_params = {'iterations': [200,300,400,500],
        #             'depth' : [4,5,6,7,8,9, 10],
        #             'learning_rate' : [0.01,0.02,0.03,0.04,0.1],
        #             'iterations'    : [50, 100,150,200,300]}
        cv_params = {
            'depth' : [2,3,4],
            'learning_rate' : [0.01,0.02,0.03],
            'iterations'    : [100]}
        other_params = {
            'iterations': 1000,
            # 'learning_rate':0.03,
            'learning_rate':0.05,
            'l2_leaf_reg':3,
            'bagging_temperature':1,
            'random_strength':1,
            'depth':6,
            'rsm':1,
            'one_hot_max_size':2,
            'leaf_estimation_method':'Gradient',
            'fold_len_multiplier':2,
            'border_count':128,
        }
        model_cb = CatBoostRegressor(**other_params)
        # model_cb = CatBoostRegressor()
        # optimized_CATM = GridSearchCV(estimator=model_cb, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
        # optimized_CATM.fit(X_train,y_train)#,cat_features =category_features)

        # test_score = optimized_CATM.score(X_test,y_test)
        # print('test 得分:{0}'.format(test_score))
        # self.log.debug('%s test_score 得分:%f'  % (self.__class__.__name__ , test_score))
        # print("Best score: %0.3f" % optimized_CATM.best_score_)
        # print('参数的最佳取值：{0}'.format(optimized_CATM.best_params_))
        # self.log.debug('%s best_params_ {0}'.format(optimized_CATM.best_params_)  % self.__class__.__name__ )
        # print('最佳模型得分:{0}'.format(optimized_CATM.best_score_))


        # # print(optimized_CATM.cv_results_['mean_test_score'])
        # # print(optimized_CATM.cv_results_['params'])
        # bst_model = optimized_CATM.estimator
        # print(bst_model)
        bst_model =model_cb
        bst_model.fit(X,y)
        # # 效果评估，均方误差,均方根误差，R2
        # y_predict=bst_model.predict(X_test)
        # r2=round(r2_score(y_test, y_predict),4)

        return bst_model