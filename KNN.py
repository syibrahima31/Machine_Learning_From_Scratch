import numpy as np 
import pandas as pd 
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from  collections import Counter



#import the dataset from api in sklearn
iris = datasets.load_iris()
X, y = iris.data, iris.target

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size =0.18, random_state=49)


class Knn:
    def __init__(self, k ):
        self.k = k

    def fit(self, x_train, y_train):
        self.X = x_train
        self.y = y_train
         


    def predict(self, x_test):
        return np.array([self._predict(x) for x in x_test])

    def euclid_distance(self,vect_1, vect_2):
        return np.sqrt(np.sum( (vect_1-vect_2)**2))



    def _predict(self,obs):
        distance = [self.euclid_distance(x, obs) for x in self.X]
        index = np.argsort(distance)
        labels = self.y[index]
        k_labels = labels[:self.k]
        count = Counter(k_labels).most_common()
        classe = count[0][0]
        return classe



    def score(self, x_test, y_test):
        prediction = self.predict(x_test)
        acc = np.sum(prediction ==y_test )
        return acc / x_test.shape[0]    




model = Knn(k=3)
model.fit(x_train,y_train)
prediction = model.predict(x_test)
model.score(x_test, y_test)

# print accuracy 

def evaluate():
    acc_test  = []
    acc_train = []
    for i in range(1, 202):
        model = Knn(k=i)
        model.fit(x_train,y_train)
        score_test = model.score(x_test, y_test)
        score_train = model.score(x_train, y_train)
        acc_test.append(score_test)
        acc_train.append(score_train)

    plt.figure()
    plt.plot(range(1, 202), acc_test, label="test set")
    plt.plot(range(1, 202), acc_train, label="train set")
    plt.legend()
    plt.show() 

evaluate()

