import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

class SVM:
    def __init__(self,lr=0.001,n_iter=1000,lambda_par=0.01):
        self.lr = lr
        self.n_iters = n_iter
        self.lambda_par = lambda_par
        self.w = None
        self.b = None
    
    def fit(self,X,y):
        self.w = np.zeros(X.shape[1])
        self.b = 0

        ## change labels of y=0 to y=-1 
        y = np.where(y==0,-1,y)
 
        for it in range(self.n_iters):
            for i,x in enumerate(X):
                cond = y[i] * (np.dot(x,self.w) - self.b) ## calculating for each point
                if cond>=1:
                    self.w -= self.lr * 2 * self.lambda_par * self.w
                else:
                    self.w -= self.lr * (2 * self.lambda_par * self.w  -(x * y[i]))
                    self.b -= self.lr * y[i]

    def predict(self,X):
        pred = np.dot(X,self.w) - self.b
        return np.where(pred<=0,-1,1)
    

    # Testing
if __name__ == "__main__":
    X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
    y = np.where(y == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    clf = SVM()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    print("SVM classification accuracy", accuracy(y_test, predictions))

    def visualize_svm():
        def get_hyperplane_value(x, w, b, offset):
            return (-w[0] * x + b + offset) / w[1]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amax(X[:, 0])

        x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
        x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

        x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
        x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

        x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
        x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

        ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

        x1_min = np.amin(X[:, 1])
        x1_max = np.amax(X[:, 1])
        ax.set_ylim([x1_min - 3, x1_max + 3])

        plt.show()

    visualize_svm()


    