import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        self.classes = None
        self.priors = {}
        self.mean = {}
        self.var = {}
    
    def fit(self,X,y):
        self.classes = np.unique(y)
        for cls in self.classes:
            X_cls = X[y==cls]
            self.priors[cls] = X_cls.shape[0]/X.shape[0]
            self.mean[cls] = np.mean(X_cls,axis=0) ## class wise , feature wise mean and variance
            self.var[cls] = np.var(X_cls,axis=0)
    
    def predict(self,X):
        return [self._predict(x) for x in X]
    
    def _predict(self,x):
        prob = []
        for cls in self.classes:
            ## priors
            prior = np.log(self.priors[cls])
            ## posterior
            posterior = np.sum(np.log(self._pdf(x,cls)))

            ## total 
            prob.append(prior+posterior)

        return self.classes[np.argmax(prob)]

    def _pdf(self,x,cls):
        ## gaussian density function for each feature in x given class
        mean = self.mean[cls]
        var = self.var[cls]
        numerator = np.exp((-1*(x-mean)**2)/(2*var))
        denominator = np.sqrt(2*np.pi*var)
        # print("pdf")
        # print(numerator/denominator) ## length will be equal to number of features(1,num features)
        return numerator/denominator

  

