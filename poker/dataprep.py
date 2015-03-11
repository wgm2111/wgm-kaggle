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

# Read in the training set
# --
ftrain = 'train.csv'
train_data_frame = read_csv(ftrain)   # data frame

ftest = 'test.csv'
test_data_frame = read_csv(ftest)

# Make arrays out of the data
Xsc = train_data_frame.as_matrix(
    ['S1', 'C1', 'S2', 'C2','S3', 'C3','S4', 'C4','S5', 'C5'])
y = train_data_frame.as_matrix(['hand'])

# read in the test set
# --
Xsc_test = test_data_frame.as_matrix(
    ['S1', 'C1', 'S2', 'C2','S3', 'C3','S4', 'C4','S5', 'C5'])



# make an alternative form of the data 
