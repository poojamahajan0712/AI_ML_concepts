import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

class SVM:
    def __init__(self,lr=0.001,n_iter=1000,lambda_par=0.01):
        self.lr = lr
        self.n_iters = n_iter
        self.lambda_par = lambda_par
        self.w = None
        self.b = None
    
    def fit(self,X,y):
        self.w = np.zeros(X.shape[1])
        self.b = 0

        ## change labels of y=0 to y=-1 
        y = np.where(y==0,-1,y)
        
        ## when the preiction is correct i.e. y_ac * y_pr >=1 , then hard margin maximisation i.e. min w, else both hard margin + hing loss is minimised i.e max(0,1-y_ac*y_pr)
        ## here y_pr = w*x-b
        for it in range(self.n_iters):
            for i,x in enumerate(X):
                cond = y[i] * (np.dot(x,self.w) - self.b) ## calculating for each point
                if cond>=1:
                    self.w -= self.lr * 2 * self.lambda_par * self.w
                else:
                    self.w -= self.lr * (2 * self.lambda_par * self.w  -(x * y[i]))
                    self.b -= self.lr * y[i]

    def predict(self,X):
        pred = np.dot(X,self.w) - self.b
        return np.where(pred<=0,-1,1)
    

    # Testing
if __name__ == "__main__":
    X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
    y = np.where(y == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    clf = SVM()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    print("SVM classification accuracy", accuracy(y_test, predictions))

    