#Lasso
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
import itertools
def lasso(data_train, label):
    X_train = data_train.iloc[:,:]
    y = label
    alphas = np.logspace(-3,1,50)
    model_lasso = LassoCV(alphas = alphas, cv=10, max_iter=10000).fit(X_train, y)
    # print(model_lasso.alpha_)
    coef = pd.Series(model_lasso.coef_, index=X_train.columns)
    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
    index = coef[coef != 0].index
    X_lasso=X_train[index]
    return X_lasso
data_train = pd.read_excel(r'./data/xxx.xlsx')
label = pd.read_excel(r'./data/xxx.xlsx')
lasso_features = lasso(data_train, label)