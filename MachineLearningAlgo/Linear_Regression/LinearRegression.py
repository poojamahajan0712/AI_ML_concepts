import numpy as np
## Linear Regression using Gradient descent 
class LinearRegression:
    
    def __init__(self,lr=0.002,n_iters = 500):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
          
    def fit(self,X,y):
         n_samples, n_features = X.shape
         self.weights=np.zeros(n_features)
         self.bias = 0

         for _ in range(self.n_iters):
            y_pred = np.dot(X,self.weights)+self.bias ## here X is 2D i.e (samples,features) and np.dot will result in output of size n_samples , it multiplies each row's  each feature with corresponding weight and sums it to get one value for one row (dot product) 
            ## also note np.dot for 2D is like matrix multiplication so order of inputs matter, you cannot pass weights first and then X , it will result in error, while for 1D it wont matter.
            dw = (1/n_samples)* 2 *np.dot(X.T,(y_pred-y)) ## here say if X shape is (20,5) and y shape is (20,), we need to transpose the X to perform np.dot opertion and output will be 5 values where 5 corresponds to weights and features.
            db = (1/n_samples)* 2 *np.sum(y_pred-y)  ## single value

            ## the above formulas dw and db are obtained via performing partial derivatives of mean squared error of prediction and actual output.

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db
       
          
    def predict(self,X):
        return  np.dot(X,self.weights)+self.bias    
          