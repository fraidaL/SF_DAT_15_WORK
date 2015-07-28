# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 20:33:14 2015

@author: fraidylevilev
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import statsmodels.formula.api as smf

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

titanic = pd.read_csv('/Users/fraidylevilev/Desktop/GA/SF_DAT_15/data/titanic.csv')
titanic.head()

features = ['Pclass', 'Parch']
X = titanic[features]
y = titanic.Survived

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
zip(features, log_reg.coef_[0])

print log_reg.intercept_
print log_reg.coef_

log_reg.predict(X_test)

from sklearn import metrics
preds = log_reg.predict(X_test)
print metrics.confusion_matrix(y_test, preds)

df = pd.read_csv('https://raw.githubusercontent.com/sinanuozdemir/SF_DAT_15/master/data/titanic.csv')

mean_m_age = df.groupby('Sex').Age.mean().ix['male']
mean_f_age = df.groupby('Sex').Age.mean().ix['female']

def round_fare(x):
    return round(x, 2)
    
df.Fare.apply(round_fare)    

