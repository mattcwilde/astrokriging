# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 11:41:38 2016

@author: dflemin3 [David P Fleming, University of Washington]
@email: dflemin3 (at) uw (dot) edu

This script performs using xgboost (Chen and Geustrin 2016) to fit to VPLANET
simulation results.  It looks like xgboost is state of the art, so we'll see
how that goes.  The reason xgboost has its own script for fitting is that
optimizing its parameters can be quite tricky as there are many.  Here, I clamp
some parameters and optimize over the others using cross validation on a subset
of the training set.  Once the hyperparameters have been optimized, I then refit
over the entire training set and test on the testing set to see how I did.

Thankfully, xgboost has sklearn API that looks like this:

Phys best fit:

best_params = {"base_score" : 0.5, "colsample_bylevel" : 1, "colsample_bytree" : 1.0,
       "gamma" : 0.2, "learning_rate" : 0.05, "max_delta_step" : 0, "max_depth" : 8,
       "min_child_weight" : 5, "n_estimators" : 630, "nthread" : -1,
       "objective" : 'reg:linear', "reg_alpha" : 0, "reg_lambda" : 0.0,
       "scale_pos_weight" : 1, "seed" : seed, "silent" : True, "subsample" : 0.9}

Poly best fit:

XGBRegressor(base_score=0.5, colsample_bylevel=1, colsample_bytree=1.0,
       gamma=0.3, learning_rate=0.05, max_delta_step=0, max_depth=6,
       min_child_weight=5, missing=None, n_estimators=630, nthread=-1,
       objective='reg:linear', reg_alpha=0, reg_lambda=0.0,
       scale_pos_weight=1, seed=42, silent=True, subsample=0.7)

\begin{tabular}{llrrrr}
\toprule
{} &  XGBoost &  Training MSE &  Testing MSE &  Training R2 &  Testing R2 \\
\midrule
0 &  XGBoost &       0.00871 &     0.028451 &     0.963597 &    0.880698 \\

"""

# Imports
from __future__ import print_function, division, unicode_literals
import os
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import xgboost as xgb
import pandas as pd

################################################################
#
# Try with smaller Physical feature set
#
################################################################

# Flags to control functionality
poly = False
phys = True
save_fit = False
run_fit = True

# Data, cache locations
data_loc = "../Data"
phys_cache = "proc_physical_3sig.pkl"
phys_poly_cache = "proc_physical_poly_3sig.pkl"
cache_loc = "/astro/store/gradscratch/tmp/dflemin3/ML_Data"

# What cache do we want to open?
if poly and not phys:
    cache = phys_poly_cache
    xg_cache = "xg_poly_model.pkl"
elif phys and not poly:
    cache = phys_cache
    xg_cache = "xg_phys_model.pkl"
else:
    raise ValueError("Either poly or phys, not both!")

# Load data
if os.path.exists(os.path.join(data_loc,cache)):
    print("Reading data from cache:",cache)
    with open(os.path.join(data_loc,cache), 'rb') as handle:
        X, y, names = pickle.load(handle)
else:
    raise NameError("%s not defined." % cache)

print(X.shape)

################################################################
#
# Set up
#
################################################################

test_frac = 0.2
val_frac = 0.2 # Fraction of training data to use as validation for training hyperparams
k = 5 # number of folds for cross validation
seed = 42

# Split data into training set, testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac,
                                                    random_state=seed)

# Scale data to 0 mean, 1 std based on training data
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

if run_fit:

    # For cross validation, do k splits with validation on val_frac of training data held out
    cv = ShuffleSplit(n_splits=k, test_size=val_frac, random_state=seed)

    # Dict to save best parameters (default with no l1, l2 regularization)
    best_params = {"reg_lambda" : 0.0, "seed" : seed}

    ################################################################
    #
    # Step one: CV over learning rate, num estimators, fix everything
    # else
    #
    ################################################################

    params = {"learning_rate" : [0.01,0.05,0.1,0.2,0.3,0.5],
              "n_estimators" : np.logspace(1,4,6).astype(int)}


    print("Performing grid search CV over learning_rate, n_estimators")
    # Make new estimator using current best parameters
    xgbr = xgb.XGBRegressor(**best_params)
    grid = GridSearchCV(xgbr, param_grid=params, cv=cv)
    est = grid.fit(X_train, y_train).best_estimator_

    # Save best fit
    best_params["learning_rate"] = est.learning_rate
    best_params["n_estimators"] = est.n_estimators

    print("Performance check...")
    print(est)
    print(est.score(X_train, y_train))
    print(est.score(X_test, y_test))

    ################################################################
    #
    # Step two: CV over min_child_weight, max_depth fix everything
    # else
    #
    ################################################################

    params = {"max_depth" : [2,4,6,8,10,12],
              "min_child_weight" : [1,2,3,4,5,6]}

    print("Performing grid search CV over min_child_weight, max_depth")
    xgbr = xgb.XGBRegressor(**best_params)
    grid = GridSearchCV(xgbr, param_grid=params, cv=cv)
    est = grid.fit(X_train, y_train).best_estimator_

    # Save best parameters
    best_params["max_depth"] = est.max_depth
    best_params["min_child_weight"] = est.min_child_weight

    print("Performance check...")
    print(est)
    print(est.score(X_train, y_train))
    print(est.score(X_test, y_test))

    ################################################################
    #
    # Step three: CV over subsample, gamma fix everything
    # else
    #
    ################################################################

    params = {"subsample" : [0.4,0.5,0.6,0.7,0.9,1.0],
              "gamma" : [0.0,0.1,0.2,0.3,0.4,0.5]
              }

    print("Performing grid search CV over subsample, gamma")
    xgbr = xgb.XGBRegressor(**best_params)
    grid = GridSearchCV(xgbr, param_grid=params, cv=cv)
    est = grid.fit(X_train, y_train).best_estimator_

    # Save best parameters
    best_params["subsample"] = est.subsample
    best_params["gamma"] = est.gamma

    print("Performance check...")
    print(est)
    print(est.score(X_train, y_train))
    print(est.score(X_test, y_test))

    ################################################################
    #
    # Step four: CV over colsample_bytree
    # else
    #
    ################################################################

    params = {"colsample_bytree" : [0.3,0.4,0.5,0.7,0.9,1.0]}

    print("Performing grid search CV over colsample_bytree")
    xgbr = xgb.XGBRegressor(**best_params)
    grid = GridSearchCV(xgbr, param_grid=params, cv=cv)
    est = grid.fit(X_train, y_train).best_estimator_

    # Save best parameters
    best_params["colsample_bytree"] = est.subsample

    print("Performance check...")
    print(est)
    print(est.score(X_train, y_train))
    print(est.score(X_test, y_test))

################################################################
#
# Final evaluation, cache model?
#
################################################################

# Refit with optimized hyperparameters
xgbr = xgb.XGBRegressor(**best_params)
xgbr.fit(X_train, y_train)

# Cache model?
if save_fit and not os.path.exists(os.path.join(cache_loc,xg_cache)):
    print("Caching model at %s" % os.path.join(cache_loc,xg_cache))
    with open(os.path.join(cache_loc,xg_cache), 'wb') as handle:
        pickle.dump(xgbr, handle)

# Evaluate Model
train_r2 = xgbr.score(X_train, y_train)
test_r2 = xgbr.score(X_test, y_test)

y_hat_train = xgbr.predict(X_train)
train_mse = mean_squared_error(y_train, y_hat_train)

y_hat_test = xgbr.predict(X_test)
test_mse = mean_squared_error(y_test, y_hat_test)

# Save to latex style table
data = [['XGBoost',train_mse, test_mse, train_r2, test_r2]]

col_names = ['XGBoost','Training MSE','Testing MSE', 'Training R2', 'Testing R2']

table = pd.DataFrame(data=data, columns=col_names)
print(table.to_latex())
