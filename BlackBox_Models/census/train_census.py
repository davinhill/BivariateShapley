# adapted from https://github.com/slundberg/shap

import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


import pandas as pd
import numpy as np
import xgboost
from sklearn.model_selection import train_test_split
from sklearn import metrics

'''
np.random.seed(1)

X,y = shap.datasets.adult()
X_display,y_display = shap.datasets.adult(display=True)

# create a train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
d_train = xgboost.DMatrix(X_train, label=y_train)
d_test = xgboost.DMatrix(X_test, label=y_test)
X_test.to_pickle('./Data/census_x_test.pkl')
np.savetxt('./Data/census_y_test.csv', y_test, delimiter = ',')
X_train.to_pickle('./Data/census_x_train.pkl')
np.savetxt('./Data/census_y_train.csv', y_train, delimiter = ',')

'''

X_train = pd.read_pickle('../Data/census_x_train.pkl')
y_train = np.loadtxt('../Data/census_y_train.csv')

X_test = pd.read_pickle('../Data/census_x_test.pkl')
y_test = np.loadtxt('../Data/census_y_test.csv')

colnames = ['Age', 'Workclass', 'Education-Num', 'Marital Status', 'Occupation',
       'Relationship', 'Race', 'Sex', 'Capital Gain', 'Capital Loss',
       'Hours per week', 'Country']

xgb_train = xgboost.DMatrix(X_train, label=y_train)
xgb_test = xgboost.DMatrix(X_test, label=y_test)

d_train = xgboost.DMatrix(X_train, label=y_train)
d_test = xgboost.DMatrix(X_test, label=y_test)

params = {
    "eta": 0.01,
    "objective": "binary:logistic",
    "subsample": 0.5,
    "base_score": np.mean(y_train),
    "eval_metric": "logloss"
}
model = xgboost.train(params, d_train, 5000, evals = [(d_test, "test")], verbose_eval=100, early_stopping_rounds=20)

pred =(model.predict(d_test) >= 0.5)*1
# see how well we can order people by survival
metrics.accuracy_score(y_test, pred)

model.save_model('model_census.json')
print('done!')