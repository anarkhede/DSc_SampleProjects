import pandas as pd
from matplotlib import pyplot as plt

# For Red Wine
# 1) Read csv as dataframe
red_wine = pd.read_csv('/Users/atul/Desktop/ML_UCI_datasets/winequality-red.csv')

rc = red_wine.corr(method='pearson')

plt.matshow(rc)
plt.xticks(range(len(red_wine.columns)), red_wine.columns, rotation=60, ha='left')
plt.yticks(range(len(red_wine.columns)), red_wine.columns)
plt.colorbar()

# Drop one of highly corelated variables: fixed_acidity to citric_acid and density
# free_sulfur_dioxide to total_sulfur_dioxide

red_wine.drop(['citric acid', 'density', 'total sulfur dioxide'], axis=1, inplace=True)

X = red_wine.ix[:,0:red_wine.shape[1]-1]
y = red_wine.ix[:,red_wine.shape[1]-1]

# 2) Test/train split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

def plot_model_scores(model):
    s = [model.score(X_train, y_train), model.score(X_test, y_test)]
    plt.barh([0, 1], s, align='center')
    plt.yticks([0, 1], ['Train', 'Test'])
    plt.xlabel("Score")
    plt.ylabel("Training OR Test")

# 3) Fit RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
# n_estimators should be log2(n_features) for reg and sqrt(n_features) for classf
forest = RandomForestRegressor(n_estimators=3, random_state=0)
forest.fit(X_train, y_train)
plot_model_scores(forest)
print("Random Forest Train Score: ", forest.score(X_train, y_train))
print("Random Forest Test Score: ", forest.score(X_test, y_test))

# Fit Decision Tree
from sklearn.tree import DecisionTreeRegressor
t = DecisionTreeRegressor(max_depth=50, random_state=20)
t.fit(X_train, y_train)
print("Decision Tree Train Score: ", t.score(X_train, y_train))
print("Decision Tree Test Score: ", t.score(X_test, y_test))


def plot_features_importance(model):
    plt.barh(range(len(X.columns)), model.feature_importances_, align='center')
    plt.yticks(range(len(X.columns)), red_wine.columns)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")

plot_features_importance(t)

# 4) Test our model
from sklearn.model_selection import cross_val_score
scores = cross_val_score(forest, X_train, y_train, cv=5)
print("Cross-validation scores: {}".format(scores))
print("Average cross-va;idation scores: {}".format(scores.mean()))

# 5) Make Prediction
print("Random Forest Test Score: ", forest.score(X_test, y_test))




# For White Wine
# 1) Read red wine csv as dataframe
white_wine = pd.read_csv('/Users/atul/Desktop/ML_UCI_datasets/winequality-white.csv')

wc=white_wine.corr(method='pearson')

plt.matshow(wc)
plt.xticks(range(len(white_wine.columns)), white_wine.columns, rotation=60, ha='left')
plt.yticks(range(len(white_wine.columns)), white_wine.columns)
plt.colorbar()

X = white_wine.ix[:,0:white_wine.shape[1]-1]
y = white_wine.ix[:,white_wine.shape[1]-1]

# 2) Test/train split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 3) Fit Linear Regressor
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train, y_train)
print("Training LR set score: {:.2f}".format(lr.score(X_train, y_train)))

# 4) Test our model
from sklearn.model_selection import cross_val_score
scores = cross_val_score(lr, X_train, y_train, cv=5)
print("Cross-validation scores: {}".format(scores))
print("Average cross-va;idation scores: {}".format(scores.mean()))

# 5) Make Prediction
print("Random Forest Test Score: ", lr.score(X_test, y_test))