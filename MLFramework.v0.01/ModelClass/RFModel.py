from BaseClass.MLModelBase import MLModelBase
from sklearn.ensemble import RandomForestRegressor

class RFModel(MLModelBase):
    def training(self,X,y):
        rf = RandomForestRegressor(n_estimators = 30, random_state = 42)
        rf.fit(X, y) 
        return rf