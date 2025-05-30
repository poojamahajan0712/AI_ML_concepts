import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate 
        self.lambda_param = lambda_param 
        self.n_iters = n_iters 
        self.w = None 
        self.b = None  

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0 

        y_ = np.where(y <= 0, -1, 1)  

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                
                if condition:
                    # If condition is met, update weights using only regularization term
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # If condition is not met, update weights with hinge loss gradient
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx)  
