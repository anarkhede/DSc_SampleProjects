# Import Libraries:
import pandas as pd
from matplotlib import pyplot as plt

iris = pd.read_csv('/Users/atul/Documents/INTERVIEW_PREP/Study/ML/ML_data/iris.csv')
ic = iris.corr(method='pearson')

plt.matshow(ic)
plt.xticks(range(len(iris.columns)), iris.columns, rotation=60, ha='left')
plt.yticks(range(len(iris.columns)), iris.columns)
plt.colorbar()

X = iris.ix[:, 1:iris.shape[1]-1] # X is features
y = iris.ix[:, iris.shape[1]-1] # y is target

# Viz data, 4 features by targets
# plt.subplot(6,1,1)
# plt.scatter(X.ix[y=='setosa', 0], X.ix[y=='setosa', 1], color='r', marker='o')

# 2) Test/train split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


def plot_model_scores(model):
    s = [model.score(X_train, y_train), model.score(X_test, y_test)]
    plt.barh([0, 1], s, align='center')
    plt.yticks([0, 1], ['Train', 'Test'])
    plt.xlabel("Score")
    plt.ylabel("Training OR Test")


# First model: K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
plot_model_scores(knn)

# Second model: SVC
from sklearn.svm import LinearSVC
svc = LinearSVC()
svc.fit(X_train, y_train)
# Check following scores, if close then model overfitting
plot_model_scores(svc)

# Fit model: Random Forest Regressor
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=2, random_state=0)
forest.fit(X_train, y_train)
plot_model_scores(forest)



