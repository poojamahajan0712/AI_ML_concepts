import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from knn import KNN 
import matplotlib.pyplot as plt
X, y = datasets.make_regression(n_samples=200, n_features=10, noise=1, random_state=4) 
print(X[0])
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
model = KNN(k=3,is_regression=True)
model.fit(X_train,y_train)
predictions = model.predict(X_test)
# print(predictions)

def rmse(y_test,y_pred):
    return np.sqrt(np.mean((y_test-y_pred)**2))
print("rmse", np.round(rmse(y_test,predictions),2))