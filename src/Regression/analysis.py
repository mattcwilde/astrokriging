# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 11:41:38 2016

@author: dflemin3

This script performs model comparison to see which estimator performs the best on VPLANET
simulation results.  The script then performs a bootstrapping procedure on each fitted
module to see where it performs poorly and hence where additional simulations need to
be ran.

For this script, I test linear regression, ridge regression and an ensemble
method, Random Forest regression.  Error estimates computed using bootstrapping
will only work for linear and ridge regression and boostrapping is used to
build the random forest regressor.

"""
# Imports
from __future__ import print_function, division, unicode_literals
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import bootstrap_utils as bu
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge



# bigplanet Imports
from bigplanet import data_extraction as de

# Flags to control functionality
seed = 42 # RNG seed

################################################################
#
# Load dataframe from cache, exploratory data analysis
#
################################################################

data_loc = "../Data"
phys_cache = "df_physical_3sig.pkl"
phys_poly_cache = "df_physical_poly_3sig.pkl"
phys_model_cache = "physical_model.pkl"

# Load polynomial transformed data
if os.path.exists(os.path.join(data_loc,phys_poly_cache)):
    print("Reading data from cache:",phys_poly_cache)
    with open(os.path.join(data_loc,phys_poly_cache), 'rb') as handle:
        Xpoly, y, names = pickle.load(handle)
else:
    raise NameError("%s not defined." % phys_poly_cache)

# Load physical data
if os.path.exists(os.path.join(data_loc,phys_cache)):
    print("Reading data from cache:",phys_cache)
    with open(os.path.join(data_loc,phys_cache), 'rb') as handle:
        X, y, names = pickle.load(handle)
else:
    raise NameError("%s not defined." % phys_cache)

################################################################
#
# Train, test, compare!
#
################################################################

test_frac = 0.2
val_frac = 0.2 # Fraction of training data to use as validation for training hyperparams
n_alpha = 50 # Size of alpha grid search for ridge
n_C = 5
n_gamma = 5
k = 5 # number of folds for cross validation

# Split data into training set, testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac,
                                                    random_state=seed)

# Make list of model defaults
models = [LinearRegression(),
          Ridge(random_state=seed),
          RandomForestRegressor(random_state=seed)
          ]

# Do we need to train model's hyperparameters?
train_hyper = [False, True, True]

# For cross validation, do k splits with validation on val_frac of training data held out
cv = ShuffleSplit(n_splits=k, test_size=val_frac, random_state=seed)

# List of dicts of params for grid search cross validation
hyper_ranges = [{},
                {"alpha":np.logspace(-10,1,n_alpha)},
                {"max_depth":[2,4,6,8,10,None]}
                ]

# Containers for error/loss metrics
train_r2 = []
train_mse = []
test_r2 = []
test_mse = []

# Loop over models!
for ii in range(len(models)):

    # If you need to train hyper parameters
    if train_hyper[ii]:
        print("Training hyperparameters and fitting:",models[ii])

        # Run grid search on subset of training data, overwrite model with best fit
        # using k fold cross validation
        grid = GridSearchCV(models[ii], param_grid=hyper_ranges[ii], cv=cv)

        # Now refit over entire training set with best hyperparameters
        models[ii] = grid.fit(X_train, y_train).best_estimator_

        # Save training R^2, MSE for train, test set

        # Train
        y_hat_train = models[ii].predict(X_train)
        train_mse.append(mean_squared_error(y_train, y_hat_train))
        train_r2.append(models[ii].score(X_train, y_train))

        # Test
        y_hat_test = models[ii].predict(X_test)
        test_mse.append(mean_squared_error(y_test, y_hat_test))
        test_r2.append(models[ii].score(X_test, y_test))


    # No hyperparameters, just fit on training data!
    else:
        print("Fitting:",models[ii])
        models[ii].fit(X_train, y_train)

        # Save training R^2, MSE for train, test set

        # Train
        y_hat_train = models[ii].predict(X_train)
        train_mse.append(mean_squared_error(y_train, y_hat_train))
        train_r2.append(models[ii].score(X_train, y_train))

        # Test
        y_hat_test = models[ii].predict(X_test)
        test_mse.append(mean_squared_error(y_test, y_hat_test))
        test_r2.append(models[ii].score(X_test, y_test))

print("Training, testing r^2:")
print(train_r2,test_r2)

print("Training, testing MSE:")
print(train_mse,test_mse)

# Pickle the data to use with bootstrapping
# Now cache...
print("Caching data at %s" % os.path.join(data_loc,phys_cache))
with open(os.path.join(data_loc,phys_model_cache), 'wb') as handle:
    pickle.dump(models, handle)

# Save to latex style table
data = [['OLS',train_mse[0], test_mse[0], train_r2[0], test_r2[0]],
        ['RR',train_mse[1], test_mse[1], train_r2[1], test_r2[1]],
        ['RF',train_mse[2], test_mse[2], train_r2[2], test_r2[2]]]

col_names = ['est','training MSE','testing MSE', r'training R^2', r'testing R^2']

table = pd.DataFrame(data=data, columns=col_names)
print(table.to_latex())
