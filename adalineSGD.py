from random import shuffle
import numpy as np
from numpy.random import seed

class AdalineSGD(object):
    """
    Klasyfikator - perceptron
    
    Params:
        eta : float learning coeff
        n_iter : learning iterations
    Atribbutes:
        w : one-dim arr of weights
        errors : incorrect classifications in each epoch
        shuffle: boolean, if true, shuffles learning data
        random_state: int makes begginning weights and shuffles random
    """

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta=eta
        self.n_iter=n_iter
        self.w_initialized=False
        self.shuffle=shuffle
        if random_state:
            seed(random_state)

    def fit(self, X, y):
        """
        X: [n_probek, n_cech],ndarray
        y: [n_wynikow]

        zwraca:
            self : obiekt
        """
        # rgen = np.random.RandomState(self.random_seed)
        # self.w_=rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self._initialize_weights(X.shape[1])
        self.cost_=[]

        #self.errors_=[]

        for _ in range(self.n_iter):
            if self.shuffle:
                X,y=self._shuffle(X,y)
            cost=[]
            for xi, target in zip(X,y):
                cost.append(self._update_weights(xi,target))
            avg_cost=sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self,X,y):
        "using new learning data with old weights"
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0]>1:#wtf is ravel?--------------------
            for xi, target in zip(X,y):
                self._update_weights(xi,target)
        else:
            self._update_weights(X,y)
        return self
    
    def _shuffle(self, X, y):
        """shuffles learning data"""
        r=np.random.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self,m):
        """initialises weights"""
        self.w_=np.zeros(1+m)
        self.w_initialized=True

    def _update_weights(self,xi,target):
        """weights actualisation"""
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta*xi.dot(error)
        self.w_[0] += self.eta*error
        cost=0.5*error**2
        return cost

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) +self.w_[0]

    def activation(self,X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X)>= 0.0, 1, -1)