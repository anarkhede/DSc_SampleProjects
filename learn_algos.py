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

X = wmh.ix[:,1:wmh.shape[1]-1]
y = wmh.ix[:,wmh.shape[1]-1]

# 2) Test/train split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 3) Isolate prominant, non-corelated features also, check for over/under fitting
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
