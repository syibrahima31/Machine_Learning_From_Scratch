import numpy as np 
import pandas as pd 
from sklearn import datasets
import matplotlib.pyplot as plt 
from  collections import Counter



#import the dataset from api in sklearn
iris = datasets.load_iris()
data, target = iris.data, iris.target


class Knn:
    def __init__(self, k ):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.Y_train = y  


    def predict(self, X):
        pass

    def euclid_distance(self,x, y):
        return np.sqrt(np.sum((x-y)**2))



    def _predict(self,x):

        distance_x = np.array([self.euclid_distance(x_train,x) for x_train in self.X_train])
        distance_index =  np.argsort(distance_x)[: self.k+1]
        k_neighbors = [self.Y_train[index]  for index in distance_index]
        count = Counter(k_neighbors)
        pred = count.most_common()
        pred = pred[0][0]
        return pred

    def predict(self,X):
        vec_pred = [self._predict(prediction) for prediction in X]    
        return vec_pred

    def score(self, x_test, y_test):
        prediction = self.predict(x_test)
        acc = np.sum(prediction ==y_test )
        return acc / x_test.shape[0]    




model = Knn(k=3)
model.fit(data,target)
prediction = model.predict(data)
# print accuracy 

model.score(data, target)
