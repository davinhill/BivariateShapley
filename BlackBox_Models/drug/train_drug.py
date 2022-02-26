import pandas as pd
import numpy as np

import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)



data = pd.read_csv('../Data/drug_consumption.data') 
df = data[['Nicotine','Crack','Meth','Ketamine','Heroin','Cannibis','Coke','Amphet','Ecstasy','Mushrooms','LSD']]
for col in df.columns:
    df[[col]] = (df[[col]] != 'CL0')*1

#X = df[['Nicotine', 'Crack', 'Meth', 'Ketamine', 'Heroin', 'Cannibis', 'Coke', 'Amphet', 'Ecstasy', 'Mushrooms']]
X = df[['Nicotine','Crack','Meth','Heroin','Cannibis','Mushrooms']]
#X = df[['Nicotine', 'Crack', 'Meth', 'Ketamine', 'Heroin', 'Cannibis', 'Coke', 'Amphet', 'Ecstasy', 'Mushrooms']]
Y = df['LSD']

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
np.random.seed(1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
RF=RandomForestClassifier(max_features=None)
RF.fit(X_train, Y_train)


Y_pred=RF.predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))

'''
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, Y_train)
pred = logisticRegr.predict(X_test)
print(metrics.accuracy_score(Y_test, pred))
'''

import xgboost
xgb_train = xgboost.DMatrix(X_train, label=Y_train)
xgb_test = xgboost.DMatrix(X_test, label=Y_test)

d_train = xgboost.DMatrix(X_train, label=Y_train)
d_test = xgboost.DMatrix(X_test, label=Y_test)


params = {
    "eta": 0.01,
    "objective": "binary:logistic",
    "subsample": 0.5,
    "base_score": np.mean(Y_train),
    "eval_metric": "logloss"
}
model = xgboost.train(params, d_train, 10000, evals = [(d_test, "test")], verbose_eval=100, early_stopping_rounds=20)

pred =(model.predict(d_test) >= 0.5)*1
print(metrics.accuracy_score(Y_test, pred))

model.save_model('model_drug.json')


import h5py
with h5py.File('../Data/drug.h5', 'w') as hf:
    hf.create_dataset('data_test', data = X_test)
    hf.create_dataset('label_test', data = Y_test)
    hf.create_dataset('data_train', data = X_train)
    hf.create_dataset('label_train', data = Y_train)

import pickle
with open('model_drug.pkl', 'wb') as fid:
    pickle.dump(RF, fid) 

print('done!')