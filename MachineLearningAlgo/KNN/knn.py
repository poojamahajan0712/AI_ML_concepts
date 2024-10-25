import numpy as np

def euclidean_dist(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    def __init__(self,k):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self,X,y):
        self.X_train = X
        self.y_train = y
    
    def predict(self,X):
        return [ self._single_sample_predict(x) for x in X]
    
    def _single_sample_predict(self,x): ## private fxn
        ## find distance of sample from each point of training samples
        sample_distances = [euclidean_dist(x,sample) for sample in self.X_train]
        ## k closest distances
        closest_neighbours_indices = np.argsort(sample_distances)[:self.k]
        #k closest y corresponding to above k closest x train distances
        closest_y = [self.y_train[i] for i in closest_neighbours_indices]
        
        ## majority vote
        counter = {}
        for v in closest_y:
            counter[v] = counter.get(v,0)+1
        return max(counter,key=counter.get)

        

        