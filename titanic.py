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

# 2) Fit Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=3, random_state=0)
forest.fit(X_train, y_train)
print("Random Forest Train Score: ", forest.score(X_train, y_train))

# 3) Test our model
from sklearn.model_selection import cross_val_score

scores = cross_val_score(forest, X_train, y_train, cv=10)
print("Cross-validation scores: {}".format(scores))
print("Average cross-va;idation scores: {}".format(scores.mean()))

# 4) Read titanic_test.csv into a dataframe
titanic_test = pd.read_csv('/Users/atul/Documents/INTERVIEW_PREP/Study/ML/ML_data/titanic_test.csv')
titanic_test.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
titanic_test = titanic_test.dropna(axis=0)
pas_id = titanic_test.ix[:,0]
titanic_test = titanic_test.replace(to_replace=['male', 'female'], value=[1, 0])

X_test = titanic_test.ix[:, 1:titanic_test.shape[1]]

# 5) Make Prediction
predictions = forest.predict(X_test)
pred_df = pd.DataFrame.from_records({'Predictions': predictions, 'PasID': pas_id})
pred_df.to_csv('/Users/atul/Documents/INTERVIEW_PREP/Study/ML/ML_data/titanic_predictions.csv')


# I cleaned up the given training data. Used Random Forest Classifier as my model and
# used k-fold cross validation for checking for over-fitting and coming up with and
# average performance score which was 78%. I am submitting the above predictions with that confidence.
# Thanks!

