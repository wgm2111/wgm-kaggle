#!/usr/bin/env python

# Author: William G.K. Martin (wgm2111@cu where cu=columbia.edu)
# copyright (c) 2011
# liscence: BSD style


"""
This rewrite uses pandas to read data more elegantly.
"""

import numpy as np
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.cross_validation import cross_val_score

from basic_imports import test_card_array_true


# Read in the training set
# --
ftrain = 'train.csv'
train_data_frame = read_csv(ftrain)   # data frame

ftest = 'test.csv'
test_data_frame = read_csv(ftest)

fsample_submission = "sampleSubmission.csv"
sample_submission = read_csv(fsample_submission)
fsubmission = "my_submission.csv"


# Make arrays out of the data
Xtrain = train_data_frame.as_matrix(
    ['S1', 'C1', 'S2', 'C2','S3', 'C3','S4', 'C4','S5', 'C5'])
y = train_data_frame.as_matrix(['hand']).reshape(-1)

# read in the test set
# --
Xtest = test_data_frame.as_matrix(
    ['S1', 'C1', 'S2', 'C2','S3', 'C3','S4', 'C4','S5', 'C5'])

# Read the sample submission 


# permutations
# --
perm_array = np.eye(10, k=-2, dtype='int') + np.eye(10, k=8, dtype='int')
old = Xtrain
Xperms = [Xtrain,]
yperms = [y,]

for i in range(4):
    new = np.dot(old, perm_array)
    Xperms.append(new)
    yperms.append(y)
    old = new

Xperms = np.concatenate(Xperms)
yperms = np.concatenate(yperms)

# Transform the problem
def imbed52(X):
    "Imbed hands into a 52 dimensional space"
    N = X.shape[0]
    Icard = 13*(X[:, ::2]-1) + X[:, 1::2]-1
    Icard = Icard + 52 * np.arange(N).reshape(N, 1)
    Xcard = np.zeros((N, 52), dtype='int')
    Xcard.flat[Icard] +=1
    return Xcard

Xcard = imbed52(Xperms)
Xcard_test = imbed52(Xtest)


# RAndom Forest classifier
rfc = RandomForestClassifier()
# print("Random Forest CV score: {0}".format(cross_val_score(rfc, Xperms, yperms)))
rfc52 = RandomForestClassifier()
N = Xtrain.shape[0]
rfc52.fit(Xcard[:N], yperms[:N])
print("Random Forest CV score: {0}".format(cross_val_score(rfc52, Xcard, yperms)))

ytest = rfc52.predict(Xcard_test)







# Nperms = 5 * Xtrain.shape[0]
# Xperms = sp.zeros((Nperms, 10))
# yperms = sp.zeros((Nperms,))
# old=Xtrain
# for i in range(4):
#     new = sp.dot(old, perm_array))
#     Xperms[i * Xtrain.shape[0], :] = new
#     yperms[i * Xtrain.shape[0], :] = y
#     old = new


    

# for i in range(4):
    
#     Xperms.append(sp.dot())
#     Xperms.append(sp.dot())
# Xbig = sp.array([
#         sp.dot(X)])
# for i in range(5):
    

# perm_array = np.concatenate([
        
#         ])


# make an alternative form of the data 
