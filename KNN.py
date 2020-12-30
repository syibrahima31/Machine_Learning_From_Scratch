import numpy as np 
import pandas  as pd 
from sklearn import datasets
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from collections import Counter


dataset = datasets.load_iris()
x, y = dataset.data, dataset.target

x_train, x_test , y_train, y_test = train_test_split(x, y , test_size =0.2, random_state=49)



class Knn:
    def __init__(self, k):
        self.k = k + 1
        
    def fit(self,x_train , y_train):
        self.x = x_train 
        self.y = y_train 


    def prediction(self, X):
        prediction = [self._predict(x) for x in X]    
        return np.array(prediction)

    def score(self, x_test, y_test):
        pred = self.prediction(x_test)
        return  (np.sum(pred==y_test) ) / x_test.shape[0]  

    def euclid_dist(self, val_1, val_2 ):
        return np.sqrt(np.sum((val_1 -val_2)**2))


    def _predict(self, obs):
        distance = [self.euclid_dist(x_train, obs) for x_train in self.x] 
        index = np.argsort(distance)
        labels = self.y [index]
        k_labels = labels[:self.k]
        count = Counter(k_labels)
        classe = count.most_common()[0][0]
        return classe




if __name__ =="main":

    L_test  = []
    L_train = []
    for i in range(1, 101):
        model = Knn(i)
        model.fit(x_train, y_train)
        score_test = model.score(x_test, y_test)
        score_train = model.score(x_train, y_train)
        L_test.append(score_test)
        L_train.append(score_train)



# plot the accuracy in training set and test set
    plt.figure()
    plt.plot(range(1,101), L_test,label = "test set")
    plt.plot(range(1,101), L_train, label = "train set")
    plt.legend()
    plt.show()


# # use the k for trainnning 

    model = Knn(34)
    model.fit(x_train, y_train)
    score_test = model.score(x_test, y_test)
    score_train = model.score(x_train, y_train)

    print(f"the accuracy on the training set {score_train}")
    print(f"the accuracy on the test_set {score_test}")
