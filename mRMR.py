#mRMR
from mrmr import mrmr_classif
import pandas as pd
data = pd.read_excel('./data/xxx.xlsx')
X_T1 = data.iloc[:,9:506]
print(X_T1.shape)
y_T1 = data.loc[:,'DFS']
selected_features_T1 = mrmr_classif(X_T1, y_T1, K = 40)
X_T1C = data.iloc[:,506:1003]
print(X_T1C.shape)
y_T1C = data.loc[:,'DFS']
selected_features_T1C = mrmr_classif(X_T1C, y_T1C, K = 40)
X_T2 = data.iloc[:,1003:]
print(X_T2.shape)
y_T2 = data.loc[:,'DFS']
selected_features_T2 = mrmr_classif(X_T2, y_T2, K = 40)