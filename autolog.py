import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from scipy.stats import skew

class AutoLogTransform(BaseEstimator,TransformerMixin):
    '''
    Automatic log transform based on Feng et al (2016)
    '''
    def __init__(self,beta=None,beta_multiplier=10):
        self.beta = beta
        self.beta_multiplier = beta_multiplier
    
    def fit(self,X:np.ndarray,y=None):
        X = np.asarray(X, dtype=float)
        if self.beta is None:
            self.beta = skew(X,axis=0)*self.beta_multiplier
        self.beta = np.where(self.beta == 0, 1e-5, self.beta)
        
        self.xmin = np.min(X,axis=0)
        self.xmax = np.max(X,axis=0)
        self.R = self.xmax - self.xmin
        return self
    
    def transform(self,X:np.ndarray):
        return self.__autolog(X)

    def __autolog(self,X:np.ndarray):
        X = np.asarray(X, dtype=float)
        for i in range(X.shape[1]):
            if self.beta[i] > 0:
                X[:,i] = np.log(X[:,i] - self.xmin[i] + self.R[i]/self.beta[i])
            elif self.beta[i] < 0:
                X[:,i] = -np.log(self.xmax[i] - X[:,i] - self.R[i]/self.beta[i])
        return X