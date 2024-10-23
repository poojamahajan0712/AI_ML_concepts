import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-1*x))
    
class LogisticRegression:
    def __init__(self,lr=0.001,n_iters=500):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self,X,y):
        n_samples,n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
             y_pred = sigmoid(np.dot(X,self.weights)+self.bias)
             dw = (1/n_samples)*2*np.dot(X.T,(y_pred-y))
             db = (1/n_samples)*2*np.sum(y_pred-y)

             dw = dw - self.lr*dw
             db = db - self.lr*db

    def predict(self,X):
        y_pred = sigmoid(np.dot(X,self.weights)+self.bias)
        return [1 if y >=0.5 else 0 for y in y_pred]
            

    