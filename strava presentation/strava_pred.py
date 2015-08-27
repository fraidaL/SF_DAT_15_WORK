# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 11:08:17 2015

@author: fraidylevilev
"""

# My access token: c2f218e5c3a9af9a3e0389d3e539e197f19f650e
# My athlete id: 9753705
"""
from stravalib.client import Client
# import the strava library

client = Client()
access_token = 'c2f218e5c3a9af9a3e0389d3e539e197f19f650e'
# Input your access token below!
client.access_token = 'c2f218e5c3a9af9a3e0389d3e539e197f19f650e'
#this is me
athlete = client.get_athlete(9753705)
print("For {id}, I now have an access token {token}".format(id = athlete, token = client.access_token))
"""
#Import libraries
import requests
import pandas as pd
import numpy as np
import time
import sys

#Set up API Request variables
#calling strava api
base_url = 'https://www.strava.com/api'
#using api to get efforts on a segment
segment_url = base_url + '/v3/segments/{0}/all_efforts'
extra_headers = {'Authorization' : 'Bearer {0}'.format(access_token)}
per_page = 200


#Define Functions
# input:  segment 
# output:  list of athletes
def get_people(segment_id, pages = 1):
    #access_token = 'c2f218e5c3a9af9a3e0389d3e539e197f19f650e'
    #extra_headers = {'Authorization' : 'Bearer {0}'.format(access_token)}
    request_to_strava = requests.get('https://www.strava.com/api/v3/segments/{0}'.format(segment_id), headers=extra_headers).json()
    effort_count = request_to_strava['effort_count']
    print effort_count
    segment_url = 'https://www.strava.com/api/v3/segments/{0}/all_efforts'.format(segment_id)
    print segment_url
    params = {}
    params['start_date_local'] = '2015-06-01T00:00:00Z'
    params['end_date_local'] = '2016-01-01T23:59:59Z'
    params['per_page'] = 200
    
    all_efforts = []
    
    for number in range(1,pages + 1):
        #print number
        params['page'] = number
        segment_request = requests.get(segment_url, params = params, headers=extra_headers).json()
        all_efforts += segment_request

    new_efforts = []
    
    for effort in all_efforts:
        #print effort['athlete']
        new_efforts.append( {
        'athlete_id': effort['athlete']['id'],
        'segment_id': segment_id,
        'avg_watts': effort.get('average_watts', -1),
        'elapsed_time': effort['elapsed_time'],
        'moving_time': effort['moving_time'],
        #'average_grade': effort['segment']['average_grade'],
        #'distance': effort['segment']['distance'],
        'elevation_range': effort['segment']['elevation_high'] - effort['segment']['elevation_low']
        })
    return pd.DataFrame(new_efforts)
    
#MAKE FINAL EFFORTS TABLE
efforts = []
for segment in segments:
    efforts_for_segment = get_people(segment, pages = 8)
    efforts.append(efforts_for_segment)
    #athletes_for_segment = get_athlete_details()
    
#puts all dataframes into one 
final_efforts = pd.concat(efforts)
    
def get_segment_details(a_segment):
    #segment_results = []
    r = requests.get(base_url + '/v3/segments/{0}'.format(a_segment), headers=extra_headers)
    results = r.json()
    segment_results = {'segment_id': a_segment,
                        'seg_name': results['name'],
                       'seg_city': results['city'],
                       'avg_grade': results['average_grade'],
                       'distance': results['distance'],
                       #'elev_gain': results['total_elevation_gain']
                       }
    print r.status_code
    return segment_results
    
# getting details for specific segment
r = requests.get(base_url + '/v3/segments/{0}'.format(229781), headers=extra_headers).json()

#df_seg = pd.DataFrame(segment_results)

segments = [229781, 4313, 241885, 2371095, 2451142, 612695, 2324148, 611787, 
            652196, 688554, 620825, 632171, 650287, 613200, 1837009, 3951729,
            6486437, 132513, 2588888, 609096, 5458172, 4290971, 6993093, 
            7640621, 6010910, 6235485, 7335076, 838117, 4124323, 811512, 141491, 656860]

#MAKE FINAL SEGMENTS TABLE
segment_details = []
for segment in segments:
    details = get_segment_details(segment)
    segment_details.append(details)
    print segment_details

final_segments = pd.DataFrame(segment_details)


# get athlete details
def get_athlete_details(an_athlete):
    """ this gets all the athlete details"""
    athlete_results = []    
    url = base_url + '/v3/athletes/{0}'.format(an_athlete)
    r = requests.get(url, headers=extra_headers)
    results = r.json()
    athlete_results = {'athlete_id': an_athlete,
                        'athlete_sex': results.get('sex'),
                       'athlete_ftp': results.get('ftp'),
                       'athlete_weight': results.get('weight')}
    return athlete_results


athletes = list(final_efforts.athlete_id.unique())
athlete_details = [get_athlete_details(athlete) for athlete in athletes]
athlete_details = pd.DataFrame(athlete_details)


#correcting the problem of 0 elevation
#final_efforts[final_efforts.elevation_range == 0]
#final_segments.loc[final_segments[final_segments['segment_id'] == 2451142].index,'elev_gain'] = 308
#final_segments.loc[final_segments[final_segments['segment_id'] == 612695].index, 'elev_gain'] = 1496
#final_segments.loc[final_segments[final_segments['segment_id'] == 1837009].index, 'elev_gain'] = 21

#deleting double columns
#del final['average_grade']
#del final['distance_x']
#del final['elev_gain']

effort_segment = pd.merge(final_efforts,final_segments,how='left',left_on='segment_id',right_on='segment_id')
#MERGE ATHLETE DETAILS
#Use pd.merge to merge final and athlete_details on thhe athlete id key
final = pd.merge(effort_segment, athlete_details, how = 'left', left_on = 'athlete_id', right_on = 'athlete_id')

#renaming columns
#final = final.rename(columns = {'distance_y': 'distance'})

#adding a column to test if the segment is a climb
# take in x, a decimal
# output the type of segment based on the decimal
# just based on grade
"""
def type_of_segment(x):
    if x >= 5:
        return 'climb'
    elif x > 1 and x < 5:
        return 'bump'
    elif x >= 0 and x <= 1:
        return 'flat'
    else:
        return 'descent'
"""
        
# classifying segments based on distance, elevation, and grade
def cat_egorizer(nums):
    if nums > 80000:
        return 'HC'
    elif nums > 64000:
        return 'Cat 1'
    elif nums > 32000:
        return 'Cat 2'
    elif nums > 16000:
        return 'Cat 3'
    elif nums > 8000:
        return 'Cat 4'
    elif nums > 1500:
        return 'bump'
    elif nums >= 0:
        return 'flat'
    elif nums < 0:
        return 'descent'
        
final['type_of_segment'] = (final['avg_grade'] * final['distance']).apply(cat_egorizer)        
final.to_csv('strava.csv')

#how many times an athlete appears in df
counts = pd.DataFrame(final.athlete_id.value_counts().reset_index())
counts.columns = ['athlete_id', 'effort_count']
#search for my ID
counts[counts.athlete_id == 9753705]

final.moving_time.mean()

#linear regression
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics

strava = pd.read_csv('/Users/fraidylevilev/Desktop/GA/SF_DAT_15_WORK/strava presentation/strava.csv', index_col=0)
cat_dummies = pd.get_dummies(strava['type_of_segment'])
strava = pd.merge(strava,cat_dummies,left_index=True,right_index=True,how="left")
strava.drop_duplicates(inplace = True)

#import seaborn as sns
#sns.pairplot(strava, x_vars=['athlete_id', 'segment_id'], y_vars='moving_time', size=4.5, aspect=0.7)

#import matplotlib.pyplot as plt
#fig, axs = plt.subplots(1, 3, sharey=True)
#data.plot(kind='scatter', x='athlete_id', y='moving_time', ax=axs[0], figsize=(16, 6))
#data.plot(kind='scatter', x='segment_id', y='moving_time', ax=axs[1])
#data.plot(kind='scatter', x='avg_grade', y='moving_time', ax=axs[2])

import statsmodels.formula.api as smf
lm = smf.ols(formula ='moving_time ~ avg_watts + avg_grade + distance + elevation_range + type_of_segment', data = strava).fit()
lm.params
lm.pvalues

#replacing non-power rides with average power data so it's a more accurate prediction
my_average = strava[strava['avg_watts'] != -1]['avg_watts'].mean()
strava.avg_watts.replace(-1, my_average, inplace = True)
strava.avg_watts.replace(0, my_average, inplace = True)

strava['athlete_sex'].fillna(value = 'NA', inplace = True)
strava['athlete_sex'].replace({'M': 0, 'F': 1}, inplace = True)

#Segment Dummies
segment_dummies  = pd.get_dummies(strava.segment_id)
segment_dummy_cols = list(segment_dummies.columns)
strava = pd.merge(strava,segment_dummies,left_index=True,right_index=True,how='left')

feature_cols = ['avg_watts', 'avg_grade', 'distance', 'elevation_range'] + list(cat_dummies.columns) + segment_dummy_cols
 
#feature_cols = ['avg_watts', 'avg_grade', 'distance', 'elevation_range'] + list(cat_dummies.columns)
               
X = strava[feature_cols]
y = strava['moving_time']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
linreg = LinearRegression()
linreg.fit(X_train, y_train)
zip(feature_cols, linreg.coef_)
y_true = y_test
y_pred = linreg.predict(X_test)

# zip tells me that as watts go up, time goes down because less effort, and
# as grade goes up time goes up BY A LOT
# distance isn't much of a factor and elevation range counts a little bit

print metrics.mean_absolute_error(y_true, y_pred)
print metrics.mean_squared_error(y_true, y_pred)
print np.sqrt(metrics.mean_squared_error(y_true, y_pred))

# **Mean Absolute Error** (MAE) is the mean of the absolute value of the errors:
# $$\frac 1n\sum_{i=1}^n|y_i-\hat{y}_i|$$
# print metrics.mean_absolute_error(y_true, y_pred)

# **Mean Squared Error** (MSE) is the mean of the squared errors:
# $$\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2$$
# print metrics.mean_squared_error(y_true, y_pred)

# **Root Mean Squared Error** (RMSE) is the square root of the mean of the squared errors:
# $$\sqrt{\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2}$$
# print np.sqrt(metrics.mean_squared_error(y_true, y_pred))

# Comparing these metrics:
# - **MAE** is the easiest to understand, because it's the average error.
# - **MSE** is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
# - **RMSE** is even more popular than MSE, because RMSE is interpretable in the "y" units.

# All of these are **loss functions**, because we want to minimize them.
"""
def train_test_rmse(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))
"""

# graphs
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(strava, x_vars=['avg_grade','elevation_range','avg_watts', 'distance', 'HC', 'descent', 'flat', 'Cat 4'], y_vars='moving_time', size=4.5, aspect=0.7, kind='reg')
# the next 2 don't work
sns.pairplot(strava)
pd.scatter_matrix(strava, figsize=(12, 10))

strava.corr()
sns.heatmap(strava.corr())

strava.groupby('type_of_segment').moving_time.mean().plot(kind = 'bar', title = 'Segment type vs Time (sec)')

# decision tree
from sklearn import tree
p = strava[['moving_time','avg_watts', 'elevation_range', 'avg_grade', 'distance', 'athlete_ftp', 'athlete_sex', 'athlete_weight']+ list(cat_dummies.columns)]

time = p['moving_time']
del p['moving_time']

X_train, X_test, y_train, y_test = train_test_split(p, time, random_state=1)

# Create a decision tree classifier instance (start out with a small tree for interpretability)
ctree = tree.DecisionTreeClassifier(random_state=1, max_depth=2)

# Fit the decision tree classifier
ctree.fit(X_train, y_train)


# Create a feature vector
features = X_train.columns.tolist()

features

# How to interpret the diagram?
ctree.classes_

# Predict what will happen for 1st class woman
features
ctree.predict_proba([221, 387, 25, 1829.73, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
ctree.predict([1, 1, 25, 0, 0, 0])

# Which features are the most important?
ctree.feature_importances_

# Clean up the output
pd.DataFrame(zip(features, ctree.feature_importances_)).sort_index(by=1, ascending=False)

# Make predictions on the test set
preds = ctree.predict(X_test)

# Calculate accuracy
metrics.accuracy_score(y_test, preds)

# Confusion matrix
pd.crosstab(y_test, preds, rownames=['actual'], colnames=['predicted'])

# Make predictions on the test set using predict_proba
probs = ctree.predict_proba(X_test)[:,1]

# Calculate the AUC metric
metrics.roc_auc_score(y_test, probs)

# Decision Trees have notorouisly high variance, so what can we do
# to better estimate the out of sample error of a high variance model?





# THAT'S RIGHT! CROSS VALIDATION

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
ctree = tree.DecisionTreeClassifier(random_state=1, max_depth=2)

# compare AUC using cross-validation
from sklearn.cross_validation import cross_val_score
cross_val_score(logreg, d, survived, cv=10, scoring='roc_auc').mean()
cross_val_score(ctree, d, survived, cv=10, scoring='roc_auc').mean()





# Let's let KNN and Naive Bayes join the party




#### EXERCISE  ####
# Import the naive bayes and KNN modules and calculate an ROC/AUC score for
# each one




from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
from sklearn.neighbors import KNeighborsClassifier  # import class
knn = KNeighborsClassifier(n_neighbors=5)           # instantiate the estimator

cross_val_score(nb, d, survived, cv=10, scoring='roc_auc').mean()
cross_val_score(knn, d, survived, cv=10, scoring='roc_auc').mean()

# so far logistic regression is winning..

'''

FINE-TUNING THE TREE

'''
from sklearn.grid_search import GridSearchCV


# check CV score for max depth = 3
ctree = tree.DecisionTreeClassifier(max_depth=3)
np.mean(cross_val_score(ctree, d, survived, cv=5, scoring='roc_auc'))

# check CV score for max depth = 10
ctree = tree.DecisionTreeClassifier(max_depth=10)
np.mean(cross_val_score(ctree, d, survived, cv=5, scoring='roc_auc'))



# Conduct a grid search for the best tree depth
ctree = tree.DecisionTreeClassifier(random_state=1)
depth_range = range(1, 20)
param_grid = dict(max_depth=depth_range)
grid = GridSearchCV(ctree, param_grid, cv=5, scoring='roc_auc')
grid.fit(d, survived)


# Check out the scores of the grid search
grid_mean_scores = [result[1] for result in grid.grid_scores_]


# Plot the results of the grid search
plt.figure()
plt.plot(depth_range, grid_mean_scores)
plt.hold(True)
plt.grid(True)
plt.plot(grid.best_params_['max_depth'], grid.best_score_, 'ro', markersize=12, markeredgewidth=1.5,
         markerfacecolor='None', markeredgecolor='r')

# Get the best estimator
best = grid.best_estimator_

cross_val_score(best, d, survived, cv=10, scoring='roc_auc').mean()
cross_val_score(logreg, d, survived, cv=10, scoring='roc_auc').mean()


# Hmmm still not as good as Logistic Regression.. 
# Let's try something else



### EXERCISE ###
''' Use Grid Search try scan over three parameters
1. max_depth:     from 1 to 20
2. criterion:     (either 'gini' or 'entropy')
3. max_features : range (1,5)

'''





# Conduct a grid search for the best tree depth
ctree = tree.DecisionTreeClassifier(random_state=1)
depth_range = range(1, 20)
criterion_range = ['gini', 'entropy']
max_feaure_range = range(1,5)
param_grid = dict(max_depth=depth_range, criterion=criterion_range, max_features=max_feaure_range)
grid = GridSearchCV(ctree, param_grid, cv=5, scoring='roc_auc')
grid.fit(d, survived)

# Check out the scores of the grid search
grid_mean_scores = [result[1] for result in grid.grid_scores_]

# Get the best estimator



best = grid.best_estimator_




'''
calculate a cross-validated roc_auc score for the model and compare to 
# base logistic regression
'''

cross_val_score(best, d, survived, cv=10, scoring='roc_auc').mean()
cross_val_score(logreg, d, survived, cv=10, scoring='roc_auc').mean()


# take that logistic regression! Pew Pew!

# Decision trees (like many other classification models)
# can also be used for regression!


drinks = pd.read_csv('../data/drinks.csv', na_filter=False)

drinks

# Make dummy columns for each of the 6 regions
for continent_ in ['AS', 'NA', 'EU', 'AF', 'SA', 'OC']:
    drinks[continent_] = drinks['continent'] == continent_

drinks


del drinks['continent']
del drinks['country']
del drinks['total_litres_of_pure_alcohol'] # this doesn't seem fair does it?

X = drinks.drop('wine_servings', axis=1)
y = drinks['wine_servings']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)


rtree = tree.DecisionTreeRegressor()

rtree.fit(X_train, y_train)
rtree.predict(X_test)

scores = cross_val_score(rtree, X, y, cv=10, scoring='mean_squared_error')
mse_scores = -scores
mse_scores
rmse_scores = np.sqrt(mse_scores)
rmse_scores
rmse_scores.mean()

wine_mean = y.mean()
wine_mean

features = X.columns
pd.DataFrame(zip(features, rtree.feature_importances_)).sort_index(by=1, ascending=False)
