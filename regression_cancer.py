import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

#print(cancer.keys())  # Shows fields in the dataset
#print(cancer.target_names)
#print(cancer.feature_names)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)

# ModelI: KNearest Regressors
#from sklearn.neighbors import KNeighborsRegressor
#clf = KNeighborsRegressor(n_neighbors=3)
#clf.fit(X_train, y_train)

# If the train, test scores (or R square values) are close the model is likely under fitted.
#print("Cancer KNRegressor Train Score: ", clf.score(X_train, y_train))
#print("Cancer KNRegressor Test Score: ", clf.score(X_test, y_test))

# Model II: Ridge Higher alpha value forces coeffs closer to zero
#from sklearn.linear_model import Ridge

#ridge1 = Ridge(alpha=1).fit(X_train, y_train) # default alpha value
#print("Cancer Ridge, alpha 0, Train Score: ", ridge1.score(X_train, y_train))
#print("Cancer Ridge, alpha 0, Test Score: ", ridge1.score(X_test, y_test))

#ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
#print("Cancer Ridge, alpha 0.1, Train Score: ", ridge01.score(X_train, y_train))
#print("Cancer Ridge, alpha 0.1, Test Score: ", ridge01.score(X_test, y_test))

#ridge10 = Ridge(alpha=10).fit(X_train, y_train)
#print("Cancer Ridge, alpha 10, Train Score: ", ridge10.score(X_train, y_train))
#print("Cancer Ridge, alpha 10, Test Score: ", ridge10.score(X_test, y_test))

#  Model III: Linear Regression
#from sklearn.linear_model import LinearRegression

#lr = LinearRegression().fit(X_train, y_train)
#print("Cancer Linear Regression Train Score: ", lr.score(X_train, y_train))
#print("Cancer Linear Regression Test Score: ", lr.score(X_test, y_test))

# Plot coeffs to compare
#plt.plot(ridge1.coef_, 's', label="Ridge alpha=1")
#plt.plot(ridge01.coef_, '^', label="Ridge alpha=01")
#plt.plot(ridge10.coef_, 'v', label="Ridge alpha=10")
#plt.plot(lr.coef_, 'o', label="LinearRegression")
#plt.show()

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
t = DecisionTreeClassifier(max_depth=4, random_state=0)
t.fit(X_train, y_train)
print("Cancer Decision Tree Train Score: ", t.score(X_train, y_train))
print("Cancer Decision Tree Test Score: ", t.score(X_test, y_test))

# RANDOM FOREST


