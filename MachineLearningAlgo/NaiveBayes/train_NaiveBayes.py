from sklearn.model_selection import train_test_split
from sklearn import datasets
from Naivebayes import NaiveBayesClassifier
import numpy as np

def accuracy(y_test,y_pred):
    return np.sum(y_test==y_pred)/len(y_test) 
X, y = datasets.make_classification(n_samples=500, n_features=5, n_classes=2, random_state=123)
print("X",X.shape)
print("y",y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

nb = NaiveBayesClassifier()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)

print("accuracy",accuracy(y_test,predictions))




