import numpy as np

print("Hello World")

class Perceptron(object):
    """
    Klasyfikator - perceptron
    
    Params:
        eta : float learning coeff
        n_iter : learning iterations
    Atribbutes:
        w : one-dim arr of weights
        errors : incorrect classifications in each epoch
    """

    def __init__(self, eta=0.01, n_iter=10, random_seed=1):
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

        self.errors_=[]

        for _ in range(self.n_iter):
            errors=0
            for xi, target in zip(X,y):
                update=self.eta * (target - self.predict(xi))
                self.w_[1:] +=update*xi
                self.w_[0] +=update
                errors+=int(update!=0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) +self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X)>= 0.0, 1, -1)

                


