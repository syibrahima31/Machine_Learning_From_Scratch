import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.model_selection import train_test_split


# import data 
X, y = datasets.make_regression(n_samples=1000, n_targets =1 , n_features=1,noise =100)

x_train,x_test, y_train, y_test = train_test_split(X,y) 


plt.figure()
plt.plot(x_train, y_train, ".")
plt.plot(x_train, pred.reshape(-1,1))
plt.show()

class LinearRegression:
    def __init__(self, n_iters, lr =0.1):
        self.lr = lr 
        self.n_iters = n_iters

    def fit(self, x_train, y_train):

        self.x_train = x_train 
        self.y_train = y_train
        n_sample = self.x_train.shape[0]
        
        # define the model 
        X = np.concatenate(( np.ones(self.x_train.shape[0]).reshape(-1,1),self.x_train), axis =1)
        self.weights = np.random.rand(X.shape[1])
        self.y_pred = np.dot(X, self.weights)

        # difine loss function 

        loss = (1 /(2*n_sample)) * np.sqrt(np.sum( (self.y_train-self.y_pred)**2))

        # descent gradient
        for i in range(self.n_iters):

            gradient = (1/ n_sample) * self.x_train.T.dot(self.y_pred-self.y_train)
            self.weights -= self.lr * gradient
            self.y_pred = np.dot(X, self.weights)
        
    
        # loss function
    def predict(self, x_test):
        if self.weights.shape[0] -1 == x_test.shape[1]:
            x_test =  np.concatenate(( np.ones(x_test.shape[0]).reshape(-1,1), x_test), axis =1)
            return x_test.dot(self.weights)
        else :

            print("show your dimension")

model = LinearRegression(1000, 0.1)
weights =model.fit(x_train, y_train)
pred = model.predict(x_train)
# print (weights)

