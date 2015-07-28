# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 19:02:20 2015

@author: fraidylevilev
"""

# 2-3 classification models and one linear regression and one logistic regression

import pandas as pd
import sklearn as sklearn


data = pd.read_csv('https://raw.githubusercontent.com/sinanuozdemir/SF_DAT_15/master/data/ZYX_prices.csv')

data['1minavg'] = data['ZYX1MinSentiment'] / data['ZYX1MinTweets']
data['5minavg'] = data['ZYX5minSentiment'] / data['ZYX5minTweets']
data['10minavg'] = data['ZYX10minSentiment'] / data['ZYX10minTweets']
data['20minavg'] = data['ZYX20minSentiment'] / data['ZYX20minTweets']
data['30minavg'] = data['ZYX30minSentiment'] / data['ZYX30minTweets']
data['60minavg'] = data['ZYX60minSentiment'] / data['ZYX60minTweets']


feature_cols = ['ZYX60minSentiment']
X = data[feature_cols]
y = data['60fret']

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
zip(feature_cols, logreg.coef_[0])

# TASK 5: make predictions on testing set and calculate accuracy
y_pred_class = logreg.predict(X_test)
from sklearn import metrics
print metrics.accuracy_score(y_test, y_pred_class)


from sklearn import metrics
prds = logreg.predict(X)
#sensitivity ie TRUE POSITIVES
print metrics.confusion_matrix(y_test, y_pred_class)[1,1] / float(metrics.confusion_matrix(y_test, y_pred_class)[1,1] + metrics.confusion_matrix(y_test, y_pred_class)[1,0])
#specificity ie TRUE NEGATIVE
print metrics.confusion_matrix(y_test, y_pred_class)[0,0] / float(metrics.confusion_matrix(y_test, y_pred_class)[0,1] + metrics.confusion_matrix(y_test, y_pred_class)[0,0])