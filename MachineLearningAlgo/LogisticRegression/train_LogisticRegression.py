from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from LogisticRegression import LogisticRegression

# Load the Breast Cancer dataset has binary labels
data = datasets.load_breast_cancer()
# print(data)
X = data['data']
y = data['target']
print(X.shape) #569,30
print(y.shape) #569
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target  # 1 for malignant, 0 for benign
print(df.columns)

print(df.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=421)

model = LogisticRegression(lr=0.03,n_iters=1000)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

def accuracy(y_pred,y_test):
     return (np.sum(y_pred==y_test)/len(y_test))

print("acc", accuracy(y_pred,y_test))
