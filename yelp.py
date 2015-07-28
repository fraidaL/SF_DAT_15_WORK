# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:06:51 2015

@author: fraidylevilev
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import statsmodels.formula.api as smf

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

yelp = pd.read_csv('/Users/fraidylevilev/Desktop/GA/SF_DAT_15/hw/optional/yelp.csv')

yelp.head()
yelp.shape[0]
yelp.business_id.unique().shape[0]
#note: only 4174 unique businesses

yelp.cool
yelp.useful
yelp.funny
yelp.stars

yelp[['cool', 'stars']]

#normalize data
yelp['cool_normalized'] = yelp.cool / yelp.cool.max()
yelp['funny_normalized'] = yelp.funny / yelp.funny.max()
yelp['useful_normalized'] = yelp.useful / yelp.useful.max()

sns.pairplot(yelp, x_vars = ['cool_normalized', 'funny_normalized', 'useful_normalized'], y_vars = 'stars', size = 4.5, aspect = 0.7)
sns.pairplot(yelp, x_vars = ['cool_normalized', 'funny_normalized', 'useful_normalized'], y_vars = 'stars', size = 4.5, aspect = 0.7, kind = 'reg')

#not normalized
sns.pairplot(yelp[yelp.cool < 10], x_vars=['cool','funny','useful'], y_vars='stars', size=4.5, aspect=0.7)
yelp.plot(kind='scatter', x='cool', y='stars')
yelp.cool.hist()

#correlation matrix for all data
yelp.corr()
sns.heatmap(yelp.corr())

feature = ['cool']
X = yelp[feature]
y = yelp.stars

linreg = LinearRegression()
linreg.fit(X, y)

print linreg.intercept_
print linreg.coef_

feature = ['cool', 'useful', 'funny']
X = yelp[feature]
y = yelp.stars

linreg = LinearRegression()
linreg.fit(X, y)

print linreg.intercept_
print linreg.coef_