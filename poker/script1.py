# Script for playing with imported data



# Standard imports
from __future__ import print_function

# Numpy
import numpy as np
from sklearn import svm
from sklearn.ensemble import  (
    RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier)
from sklearn.cross_validation import cross_val_score

# data 
from basic_imports import (
    true_predictor, test_card_array_true,
    train_data, test_data, 
    training_card_array, training_value_array,
    test_card_array)



# Solve the problem with my 52 element representation
# --
X0 = train_data[:,:-1]
y0 = train_data[:,-1]
    
# Make a clasifier and train it
rf413 = RandomForestClassifier(n_estimators=52, max_depth=None)
rf413.fit(X0, y0)
rf413_score = cross_val_score(rf413, X0, y0)
print("rf413_score.mean() = {0}".format(rf413_score.mean()))



# Solve the problem with my 52 element representation
# --
X = training_card_array
y = training_value_array
# X = (X - .5) * 2



# RandomForest
# --
rf52 = RandomForestClassifier(n_estimators=52, max_depth=None)
rf52.fit(X, y)
rf52_score = cross_val_score(rf52, X, y)
print("rf52_score.mean() = {0}".format(rf52_score.mean()))

# rfex52 = ExtraTreesClassifier(n_estimators=13)
# rfex52.fit(X, y)
# rfex52_score = cross_val_score(rfex52, X, y)
# print("rfex52_score.mean() = {0}".format(rfex52_score.mean()))

# rfad52 = ExtraTreesClassifier(n_estimators=13)
# rfad52.fit(X, y)
# rfad52_score = cross_val_score(rfad52, X, y)
# print("rfad52_score.mean() = {0}".format(rfad52_score.mean()))


