from sklearn.linear_model import LinearRegression
from BaseClass.MLModelBase import MLModelBase

class LRModel(MLModelBase):
    def training(self,X,y):
        lm = LinearRegression()
        lm.fit(X, y)
        return lm