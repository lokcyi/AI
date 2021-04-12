from BaseClass.MLModelBase import MLModelBase
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
import xgboost as xgb 
from xgboost import plot_importance 

class XG(MLModelBase):
    def training(self,X,y):
        #拆分train validation set
        X_train, X_test,y_train,y_test = train_test_split(X,y,test_size =0.2,random_state=587)
        cv_params = {'n_estimators': [300,400,500,600],'max_depth':[7,11,13,15,17],
            'min_child_weight':[1,3,5,7,9]}
        other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5,
            'min_child_weight': 1, 'seed': 0,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
        model = xgb.XGBRegressor(**other_params)
        optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2'
            , cv=5, verbose=1, n_jobs=4)
        optimized_GBM.fit(X_train, y_train) 
        test_score = optimized_GBM.score(X_test,y_test)
        print('test 得分:{0}'.format(test_score))
        # evalute_result = optimized_GBM.grid_scores_
        # print('每輪迭代執行結果:{0}'.format(evalute_result))
        print('引數的最佳取值：{0}'.format(optimized_GBM.best_params_))
        print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
        #print('cv_results_',optimized_GBM.cv_results_)
        bst_model = optimized_GBM.estimator
        print(bst_model)
        bst_model.fit(X_train, y_train)
        return bst_model