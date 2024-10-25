from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from knn import KNN

iris = datasets.load_iris()
# df = pd.DataFrame(iris.data, columns=iris.feature_names)
# df['target'] = iris.target
# print(df.head())
# print(df['target'].value_counts())

X, y = iris.data, iris.target
## sepal length vs sepal width
# plt.scatter(X[:,0],X[:,1],c=y)
# plt.title("Sepal length vs Sepal width")
# plt.show()

# # petal length vs petal width 
# plt.scatter(X[:,2],X[:,3],c=y)
# plt.title("Petal length vs Petal width")
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

model = KNN(k=5)
model.fit(X_train,y_train)
predictions = model.predict(X_test)

def accuracy(y_test,y_pred):
    return np.round((np.sum(y_test==y_pred)/len(y_test))*100,2)

print("accuracy",accuracy(y_test,predictions))
