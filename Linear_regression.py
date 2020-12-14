import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.model_selection import train_test_split


# import data 
X, y = datasets.make_regression(n_samples=1000, n_targets =1 , n_features=1,noise =100)

x_train,x_test, y_train, y_test = train_test_split(X,y) 


plt.figure()
plt.plot(X, y, ".")
plt.show()

class LinearRegression:
    def __init__(self, n_iters):
        self.n_iters = n_iters

    def fit(self, x_train, y_train):
        self.x_train = x_train 
        self.y_train = y_train
        
        # define the model 
        X = np.concatenate(( np.ones(self.x_train.shape[0]).reshape(-1,1),self.x_train), axis =1)
        weights = np.random.rand(X.shape[1]).reshape(-1,1)
        y_pred = np.dot(X, weights)

        # difine loss function 

        loss = np.sqrt(np.sum( (self.y_train-y_pred)**2))
        
        return y_pred,loss 
    
        # loss function


model = LinearRegression(10)
a, b = model.fit(x_train, y_train)
