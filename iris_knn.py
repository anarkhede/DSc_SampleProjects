from sklearn.datasets import load_iris

iris = load_iris()

iris.keys()  # Shows fields in the dataset

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)

# First model K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

print("Iris KNN score: ", knn.score(X_test, y_test))

