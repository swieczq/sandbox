from cProfile import label
from ssl import ALERT_DESCRIPTION_BAD_CERTIFICATE_STATUS_RESPONSE
from turtle import color
import perceptron as pc
import AdalineGD as nAdal
import adalineSGD as sgd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap

def plot_decision_regions(X,y,classifier,resolution=0.02):

    #marker generator & color map
    markers=('s','x','o','^','v')
    colors=('red','blue','lightgreen','gray','cyan')
    cmap=ListedColormap(colors[:len(np.unique(y))])

    #draw plot of decision space
    x1_min, x1_max=X[:,0].min()-1,X[:,0].max()+1
    x2_min, x2_max=X[:,1].min()-1,X[:,1].max()+1
    xx1, xx2=np.meshgrid(np.arange(x1_min,x1_max,resolution),
                        np.arange(x2_min,x2_max,resolution))
    Z=classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z=Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    #draw plot of samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],y=X[y==cl,1],alpha=0.8,c=cmap(idx),marker=markers[idx],label=cl)


df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
df.tail()
# print(df)
y=df.iloc[0:100,4].values #y has values 0-100 from 4th column (iris name)
# print(y)
y=np.where(y=='Iris-setosa',-1,1) #y = -1 when "iris setosa", else 1
# print(y)
X=df.iloc[0:100, [0,2]].values #X has values 0-100 from columns 0-2
plt.scatter(X[:50,0],X[:50,1],color='red',marker='o',label='Setosa')
plt.scatter(X[50:100,0],X[50:100,1],color='blue',marker='x',label='Versicolor')
plt.xlabel('Dl dzialki[cm]')
plt.ylabel('Dl platka[cm]')
plt.legend(loc='upper left')
plt.show()

ppn=pc.Perceptron(eta=0.1, n_iter=10)
ppn.fit(X,y)
plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='o')
plt.xlabel('Epoki')
plt.ylabel('Liczba aktualizacji')
plt.show()

plot_decision_regions(X,y,classifier=ppn,resolution=0.02)
plt.xlabel('Dl dzialki[cm]')
plt.ylabel('Dl platka[cm]')
plt.legend(loc='upper left')
plt.show()

#------------------------------------------------------------------------
#ADALINE starts here

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
adal = nAdal.AdalineGD(n_iter=10, eta=0.01).fit(X,y)
ax[0].plot(range(1,len(adal.cost_)+1), np.log10(adal.cost_), marker='o')
ax[0].set_xlabel('Epoki')
ax[0].set_ylabel('Log (suma kwadratow bledow)')
ax[0].set_title('Adaline - wspolczynnik uczenia 0.01')
ada2 = nAdal.AdalineGD(n_iter=10, eta=0.0001).fit(X,y)
ax[1].plot(range(1,len(ada2.cost_)+1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epoki')
ax[1].set_ylabel('suma kwadratow bledow')
ax[1].set_title('Adaline - wspolczynnik uczenia 0.0001')
plt.show()

#input standarisation
X_std=np.copy(X)
X_std[:,0]=(X[:,0]-X[:,0].mean()) / X[:,0].std()
X_std[:,1]=(X[:,1]-X[:,1].mean()) / X[:,1].std()


ada = nAdal.AdalineGD(n_iter=15, eta=0.01).fit(X_std,y)
plot_decision_regions(X_std,y,classifier=ada)
plt.title("Adaline - Gradient prosty")
plt.xlabel('Dlugosc dzialki [standaryzowana]')
plt.ylabel('dlugosc platka std')
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1,len(ada.cost_)+1),ada.cost_,marker='o')
plt.xlabel("Epoki")
plt.ylabel("Suma kwadratow bledow")
plt.show()

#adalineSGD-------------------

ada=sgd.AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - stochastic gradient drop')
plt.xlabel('Dlugosc dzialki [std]')
plt.ylabel('Dlugosc platka [std]')
plt.legend(loc='upper left')
plt.show()
plt.plot(range(1, len(ada.cost_)+1),ada.cost_,marker='o')
plt.xlabel('Epoki')
plt.ylabel('sredni koszt')
plt.show()