# Import Libraries:
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# 1) Read clean_WMH.xlsx into a dataframe
iris = pd.read_csv('/Users/atul/Desktop/iris.csv')

features = iris.ix[:,1:iris.shape[1]-1]
X = features[:]
target = iris.ix[:,iris.shape[1]-1]
y = target[:]

# Viz data, 4 features by targets
plt.subplot(6,1,1)
plt.scatter(X.ix[y=='setosa', 0], X.ix[y=='setosa', 1], color='r', marker='o')
plt.scatter(X.ix[y=='versicolor', 0], X.ix[y=='versicolor', 1], color='b', marker='o')
plt.scatter(X.ix[y=='virginica', 0], X.ix[y=='virginica', 1], color='g', marker='o')

plt.subplot(6,1,2)
plt.scatter(X.ix[y=='setosa', 1], X.ix[y=='setosa', 2], color='r', marker='o')
plt.scatter(X.ix[y=='versicolor', 1], X.ix[y=='versicolor', 2], color='b', marker='o')
plt.scatter(X.ix[y=='virginica', 1], X.ix[y=='virginica', 2], color='g', marker='o')

plt.subplot(6,1,3)
plt.scatter(X.ix[y=='setosa', 2], X.ix[y=='setosa', 3], color='r', marker='o')
plt.scatter(X.ix[y=='versicolor', 2], X.ix[y=='versicolor', 3], color='b', marker='o')
plt.scatter(X.ix[y=='virginica', 2], X.ix[y=='virginica', 3], color='g', marker='o')

plt.subplot(6,1,4)
plt.scatter(X.ix[y=='setosa', 3], X.ix[y=='setosa', 0], color='r', marker='o')
plt.scatter(X.ix[y=='versicolor', 3], X.ix[y=='versicolor', 0], color='b', marker='o')
plt.scatter(X.ix[y=='virginica', 3], X.ix[y=='virginica', 0], color='g', marker='o')

plt.subplot(6,1,5)
plt.scatter(X.ix[y=='setosa', 3], X.ix[y=='setosa', 1], color='r', marker='o')
plt.scatter(X.ix[y=='versicolor', 3], X.ix[y=='versicolor', 1], color='b', marker='o')
plt.scatter(X.ix[y=='virginica', 3], X.ix[y=='virginica', 1], color='g', marker='o')

plt.subplot(6,1,6)
plt.scatter(X.ix[y=='setosa', 3], X.ix[y=='setosa', 1], color='r', marker='o')
plt.scatter(X.ix[y=='versicolor', 3], X.ix[y=='versicolor', 1], color='b', marker='o')
plt.scatter(X.ix[y=='virginica', 3], X.ix[y=='virginica', 1], color='g', marker='o')


# 2) Test/train split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)

# First model K Nearest Neighbors
#from sklearn.neighbors import KNeighborsClassifier
#knn = KNeighborsClassifier(n_neighbors=1)

#knn.fit(X_train, y_train)

#print("Iris KNN score: ", knn.score(X_test, y_test))

