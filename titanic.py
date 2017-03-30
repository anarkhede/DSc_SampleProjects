import pandas as pd

# 1) Read titanic_train.csv into a dataframe
titanic_train = pd.read_csv('/Users/atul/Documents/INTERVIEW_PREP/Study/ML/ML_data/titanic_train.csv')
titanic_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
titanic_train = titanic_train.dropna(axis=0)
titanic_train = titanic_train.replace(to_replace=['male', 'female'], value=[1, 0])

# Look at Multiple Correlations:
#titanic_train.corr(method='pearson')

# Features:
X_train = titanic_train.ix[:, 1:titanic_train.shape[1]]

# Target:
y_train = titanic_train.ix[:, 0]

# 2) Fit Random Forest Regressor
from sklearn.ensemble import RandomForestClassifier
# n_estimators should be log2(n_features) for reg and sqrt(n_features) for classf
forest = RandomForestClassifier(n_estimators=3, random_state=0)
forest.fit(X_train, y_train)
print("Random Forest Train Score: ", forest.score(X_train, y_train))

# 3) Read titanic_test.csv into a dataframe
titanic_test = pd.read_csv('/Users/atul/Documents/INTERVIEW_PREP/Study/ML/ML_data/titanic_test.csv')
titanic_test.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
titanic_test = titanic_test.dropna(axis=0)
pas_id = titanic_test.ix[:,0]
titanic_test = titanic_test.replace(to_replace=['male', 'female'], value=[1, 0])

X_test = titanic_test.ix[:, 1:titanic_test.shape[1]]

# 4) Make Prediction
predictions = forest.predict(X_test)
pred_df = pd.DataFrame.from_records({'Predictions': predictions, 'PasID': pas_id})


