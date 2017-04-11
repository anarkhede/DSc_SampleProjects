# This script reads in the excel sheet from whicap output and creates ML algo
# to predict memory function using neuro-imaging biomarkers
# Following are the steps:

# Import Libraries:
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# 1) Read clean_WMH.xlsx into a dataframe
WMH_XLfile = pd.ExcelFile('/Users/atul/Documents/INTERVIEW_PREP/Study/ML/ML_data/clean_WMH.xlsx')
wmh = WMH_XLfile.parse('Sheet1')
wmh.drop(['CortexVol', 'CorticalWhiteMatterVol', 'EC_thickness_LR', 'parahippocampus_thickness_LR', 'InfParietalLobe_thickness_LR', 'ParsOpercularis_thickness_LR', 'ParsOrbitalis_thickness_LR', 'ParsTriangularis_thickness_LR', 'InferiorFrontalLobe_thickness_LR', 'InfTemporalLobe_thickness_LR', 'TemporalPole_thickness_LR', 'Precuneus_thickness_LR', 'Supramarginal_thickness_LR', 'SuperiorParietal_thickness_LR',  'CorticalSignature'], axis=1, inplace=True)
wmh = wmh.fillna(-1, axis=0)
# Col 1 = IDNUM
# Col 2-46 = No. of voxels
# Col 49 CortexVol, Col 52 CorticalWMVol
# Col 53-56 SubcorticalGrayVol, TotalGrayVol, SupreTentorialVol, IntraCranialVol
# Col 57-192 ROI FS Thickness
# Col 193-204 = LR combined thickness
# Col 206-210 = WMH measures (ROI+Total)
# What is Cortical Signature?

X = wmh.ix[:, 1:wmh.shape[1]-1]
y = wmh.ix[:, wmh.shape[1]-1]

Xc = X.corr(method='pearson')

plt.matshow(Xc)
plt.xticks(range(len(X.columns)), X.columns, rotation=60, ha='left')
plt.yticks(range(len(X.columns)), X.columns)
plt.colorbar()

# 2) Test/train split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 3) Isolate prominant, non-corelated features also, check for over/under fitting
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

