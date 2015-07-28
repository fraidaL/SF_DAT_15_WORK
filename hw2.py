##### Part 1 #####
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt

# 1. read in the yelp dataset
yelp = pd.read_csv('/Users/fraidylevilev/Desktop/GA/SF_DAT_15/hw/optional/yelp.csv')

# 2. Perform a linear regression using 
# "stars" as your response and 
# "cool", "useful", and "funny" as predictors
feature = ['cool', 'useful', 'funny']
X = yelp[feature]
y = yelp.stars

linreg = LinearRegression()
linreg.fit(X, y)

print linreg.intercept_
print linreg.coef_

X = yelp[feature]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
linreg.fit(X_train, y_train)
zip(feature, linreg.coef_[0])
y_pred = linreg.predict(X_test)

print linreg.intercept_
print linreg.coef_

# 3. Show your MAE, R_Squared and RMSE
y_true = y_test
print metrics.mean_absolute_error(y_true, y_pred)
print metrics.mean_squared_error(y_true, y_pred)
print np.sqrt(metrics.mean_squared_error(y_true, y_pred))

sns.pairplot(yelp, x_vars=['cool','useful','funny'], y_vars='stars', size=4.5, aspect=0.7, kind='reg')


# 4. Use statsmodels to show your pvalues
# for each of the three predictors
# Using a .05 confidence level, 
# Should we eliminate any of the three?
lm = smf.ols(formula='stars ~ cool + useful + funny', data = yelp).fit()
print lm.pvalues

# 5. Create a new column called "good_rating"
# this could column should be True iff stars is 4 or 5
# and False iff stars is below 4
yelp['good_rating'] = (yelp.stars == 4) | (yelp.stars == 5)

# 6. Perform a Logistic Regression using 
# "good_rating" as your response and the same
# three predictors
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
feature_cols = ['cool', 'useful', 'funny']
X = yelp[feature_cols]
y = yelp['good_rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

# 7. Show your Accuracy, Sensitivity, Specificity
# and Confusion Matrix
from sklearn import metrics
print metrics.accuracy_score(y_test, y_pred)

prds = logreg.predict(X)
print metrics.confusion_matrix(y_test, y_pred)

# 8. Perform one NEW operation of your 
# choosing to try to boost your metrics!




##### Part 2 ######

# 1. Read in the titanic data set.
titanic = pd.read_csv('/Users/fraidylevilev/Desktop/GA/SF_DAT_15/data/titanic.csv')

# 4. Create a new column called "wife" that is True
# if the name of the person contains Mrs.
# AND their SibSp is at least 1
titanic['wife'] = ('Mrs.' in titanic.Name) & (titanic.SibSp >= 1)
titanic['wife'] = (titanic.SibSp >=1) & (titanic.Name.str.contains('Mrs'))

# 5. What is the average age of a male and
# the average age of a female on board?
mean_m_age = titanic.groupby('Sex').Age.mean().ix['male']
mean_f_age = titanic.groupby('Sex').Age.mean().ix['female']

# 5. Fill in missing MALE age values with the
# average age of the remaining MALE ages
titanic.Age.fillna(mean_m_age, inplace = 'True')

# 6. Fill in missing FEMALE age values with the
# average age of the remaining FEMALE ages
titanic.Age.fillna(mean_f_age, inplace = 'True')

# 7. Perform a Logistic Regression using
# Survived as your response and age, wife
# as predictors
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
feature_cols = ['Age', 'wife']
X = titanic[feature_cols]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
y = titanic['Survived']
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

# 8. Show Accuracy, Sensitivity, Specificity and 
# Confusion matrix


# 9. now use ANY of your variables as predictors
# Still using survived as a response to boost metrics!


# 10. Show Accuracy, Sensitivity, Specificity



# REMEMBER TO USE
# TRAIN TEST SPLIT AND CROSS VALIDATION
# FOR ALL METRIC EVALUATION!!!!

