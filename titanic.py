import pandas as pd

# 1) Read titanic_train.csv into a dataframe
titanic_train = pd.read_csv('/Users/atul/Documents/INTERVIEW_PREP/Study/ML/ML_data/titanic_train.csv')
titanic_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
titanic_train = titanic_train.dropna(axis=0)
titanic_train = titanic_train.replace(to_replace=['male', 'female'], value=[1, 0])

# Look at Multiple Correlations:
titanic_train.corr(method='pearson')

# 2) Read titanic_test.csv into a dataframe
titanic_test = pd.read_csv('/Users/atul/Documents/INTERVIEW_PREP/Study/ML/ML_data/titanic_test.csv')
titanic_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
titanic_test = titanic_test.dropna(axis=0)
titanic_test = titanic_test.replace(to_replace=['male', 'female'], value=[1, 0])

# Features:
X_train = titanic_train.ix[:, 1:titanic_train.shape[1]]
X_test = titanic_test.ix[:, 1:titanic_test.shape[1]]
# Target:
y_train = titanic_train.ix[:, 0]
y_test = titanic_test.ix[:, 0]

# 2) Fit Random Forest Regressor
from sklearn.ensemble import RandomForestClassifier
# n_estimators should be log2(n_features) for reg and sqrt(n_features) for classf
forest = RandomForestClassifier(n_estimators=3, random_state=0)
forest.fit(X_train, y_train)
print("Random Forest Train Score: ", forest.score(X_train, y_train))

# 3) Make Prediction    
y_test = forest.predict(X_test)
print(accuracy_score(y_test, predictions))

print("Random Forest Test Score: ", forest.score(X_test, y_test))

