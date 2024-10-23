# Loading libraries
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression 
import matplotlib.pyplot as plt
X, y = datasets.make_regression(n_samples=100, n_features=10, noise=20, random_state=4)
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
# plt.scatter(X[:, 0], y, color = "b", marker = "o", s = 30)
# plt.show()


reg = LinearRegression(lr=0.01)
reg.fit(X_train,y_train)
predictions = reg.predict(X_test)

def mse(y_test,y_pred):
    return np.mean((y_test-y_pred)**2)
print("mse", mse(y_test,predictions))


# ## this plot makes sense with one independent variable only, 2D , if we inc the independent variable.
# y_pred_line = reg.predict(X)
# cmap = plt.get_cmap('viridis')
# fig = plt.figure(figsize=(8,6))
# m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
# m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
# plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
# plt.show()
