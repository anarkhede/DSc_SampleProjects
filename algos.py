# This script reads in the excel sheet from whicap output and creates ML algo
# to predict memory function using neuro-imaging biomarkers
# Following are the steps:

# Import Libraries:
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# 1) Read clean_WMH.xlsx into a dataframe
WMH_XLfile = pd.ExcelFile('/Users/atul/Desktop/clean_WMH.xlsx')
wmh = WMH_XLfile.parse('Sheet1')
wmh = wmh.dropna()

features = wmh.ix[:,1:wmh.shape[1]-1]
X = features[:]
target = wmh.ix[:,wmh.shape[1]-1]
y = target[:]

# 2) Test/train split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 3) Isolate prominant, non-corelated features also, check for over/under fitting

# Regularization: Each feature should have as little effect on the outcome as possible,
# in other words, a smaller slope for the fit line.

# Ridge (L2 Regularization):
#from sklearn.linear_model import Ridge
#ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
#print("Training Ridge alpha=0.1 set score: {:.2f}".format(ridge01.score(X_train, y_train)))
#print("Test set Ridge alpha=0.1 score: {:.2f}".format(ridge01.score(X_test, y_test)))

#ridge = Ridge().fit(X_train, y_train)
#print("Training Ridge alpha=1 set score: {:.2f}".format(ridge.score(X_train, y_train)))
#print("Test set Ridge alpha=1 score: {:.2f}".format(ridge.score(X_test, y_test)))

#ridge10 = Ridge(alpha=10).fit(X_train, y_train)
#print("Training Ridge alpha=10 set score: {:.2f}".format(ridge10.score(X_train, y_train)))
#print("Test set Ridge alpha=10 score: {:.2f}".format(ridge10.score(X_test, y_test)))

# Linear Regression
#from sklearn.linear_model import LinearRegression
#lr = LinearRegression().fit(X_train, y_train)
#print("Training LR set score: {:.2f}".format(lr.score(X_train, y_train)))
#print("Test LR set score: {:.2f}".format(lr.score(X_test, y_test)))

# Lasso (L1 regularization) : For WHICAP_WMH this might be better
#from sklearn.linear_model import Lasso
#lasso = Lasso().fit(X_train, y_train)
#print("Training Lasso alpha=1 set score: {:.2f}".format(lr.score(X_train, y_train)))
#print("Test Lasso aplha=1 set score: {:.2f}".format(lr.score(X_test, y_test)))
#print("Number of features used: {}".format(np.sum(lasso.coef_ !=0)))

#lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
#print("Training Lasso alpha=001 set score: {:.2f}".format(lr.score(X_train, y_train)))
#print("Test Lasso alpha=001 set score: {:.2f}".format(lr.score(X_test, y_test)))
#print("Number of features used: {}".format(np.sum(lasso.coef_ !=0)))

# Elastic Net (Combines penalties of L1 and L2): Practically works the best
#from sklearn.linear_model import ElasticNet
#en = ElasticNet().fit(X_train, y_train)
#print("Training EN set score: {:.2f}".format(en.score(X_train, y_train)))
#print("Test EN set score: {:.2f}".format(en.score(X_test, y_test)))

#en001 = ElasticNet(alpha=0.01).fit(X_train, y_train)
#print("Training EN set score: {:.2f}".format(en001.score(X_train, y_train)))
#print("Test EN set score: {:.2f}".format(en001.score(X_test, y_test)))

# Plot to compare all models:
#plt.plot(ridge01.coef_, 's', label='Ridge alpha=0.1')
#plt.plot(ridge.coef_, '^', label='Ridge alpha=1')
#plt.plot(ridge10.coef_, 'v', label='Ridge alpha=10')

#plt.plot(lr.coef_, 'o', label='LinearRegression')

#plt.xlabel('Coeffecient Index')
#plt.ylabel('Coeffecient Magnitude')
#plt.hlines(0, 0, len(lr.coef_))
#plt.ylim(-25, 25)
#plt.legend()

# Fit Decision Tree
#from sklearn.tree import DecisionTreeRegressor
#t = DecisionTreeRegressor(max_depth=50, random_state=20)
#t.fit(X_train, y_train)

# Print Model Scores:
#print("WMH Decision Tree Train Score: ", t.score(X_train, y_train))
#print("WMH Decision Tree Test Score: ", t.score(X_test, y_test))

# Visualize Decision Tree
#from sklearn.tree import export_graphviz
#export_graphviz(t, out_file='WMH_tree.dot', class_names='mem_func', impurity=False, filled=True)

# Feature importance Matrix:
#print("Feature Importance:\n{}".format(t.feature_importances_))

# Plot Coef
#def plot_coef(model):
#    n_features = X.shape[1]
#    plt.barh(range(n_features), model.coef_, align='center')
#    plt.yticks(np.arange(n_features), model.columns)
#    plt.xlabel("Feature Coef")
#    plt.ylabel("Feature")

# Plot Feature Importance:
def plot_features_importance_wmh(model):
    n_features = wmh.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), wmh.columns)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")

#plot_features_importance_wmh(t)

# Fit Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
# n_estimators should be log2(n_features) for reg and sqrt(n_features) for classf
forest = RandomForestRegressor(n_estimators=20, random_state=40)
forest.fit(X_train, y_train)
print("WMH Random Forest Train Score: ", forest.score(X_train, y_train))
print("WMH Random Forest Test Score: ", forest.score(X_test, y_test))

# Feature importance Matrix:
print("Feature Importance:\n{}".format(forest.feature_importances_))

#plot_features_importance_wmh(forest)
