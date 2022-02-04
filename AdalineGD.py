import numpy as np

class AdalineGD(object):
    """
    Klasyfikator - perceptron
    
    Params:
        eta : float learning coeff
        n_iter : learning iterations
    Atribbutes:
        w : one-dim arr of weights
        errors : incorrect classifications in each epoch
    """

    def __init__(self, eta=0.01, n_iter=50, random_seed=1):
        self.eta=eta
        self.n_iter=n_iter
        self.random_seed=random_seed

    def fit(self, X, y):
        """
        X: [n_probek, n_cech],ndarray
        y: [n_wynikow]

        zwraca:
            self : obiekt
        """
        rgen = np.random.RandomState(self.random_seed)
        self.w_=rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_=[]

        #self.errors_=[]

        for _ in range(self.n_iter):
            output=self.net_input(X)
            errors=(y-output)
            self.w_[1:]+=self.eta * X.T.dot(errors)
            self.w_[0]+=self.eta*errors.sum()
            cost=(errors**2).sum()
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) +self.w_[0]

    def activation(self,X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X)>= 0.0, 1, -1)


                


