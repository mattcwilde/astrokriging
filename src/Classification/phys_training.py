# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 11:41:38 2016

@author: dflemin3

This script performs model comparison to see which estimator performs the best on VPLANET
simulation results.  The script then performs a bootstrapping procedure on each fitted
module to see where it performs poorly and hence where additional simulations need to
be ran.

Here, we do binary classification.

\begin{tabular}{llrrrrrrr}
\toprule
{} &  Est &  Train Loss &  Test Loss &  Train R2 &   Test R2 &  Train 01 Loss &  Test 01 Loss &  Median Std \\
\midrule
0 &   LR &    2.908153 &   2.833631 &  0.915801 &  0.917959 &       0.084199 &      0.082041 &           0 \\
1 &   RF &    0.233342 &   2.246164 &  0.993244 &  0.934967 &       0.006756 &      0.065033 &           0 \\
2 &  SVM &    1.896995 &   2.850906 &  0.945077 &  0.917459 &       0.054923 &      0.082541 &           0 \\
\bottomrule
\end{tabular}


"""
# Imports
from __future__ import print_function, division, unicode_literals
import os
import sys
sys.path.append("../Utils")
import numpy as np
import pandas as pd
import pickle
import bootstrap_utils as bu
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.metrics import log_loss, zero_one_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import preprocessing

# Flags to control functionality
show_plots = False
save_models = False

# Constants
seed = 42 # RNG seed
test_frac = 0.2
val_frac = 0.2 # Fraction of training data to use as validation for training hyperparams
k = 5 # number of folds for cross validation
n_c = 25

# Locations of caches, data
data_loc = "../../Data"
phys_cache = "prox_phys_class.pkl"
phys_model_cache = "proc_phys_class_model.pkl"
cache_loc = "/astro/store/gradscratch/tmp/dflemin3/ML_Data"
plot_loc = "../../Plots"

################################################################
#
# Load dataframe from cache, exploratory data analysis
#
################################################################

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

# Split into training, testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_frac,
                                                    random_state=seed)

# Scale data to 0 mean, 1 std based on training data
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Make list of model defaults
models = [LogisticRegression(random_state=seed),
          RandomForestClassifier(random_state=seed),
          SVC(random_state=seed)]

# Do we need to train model's hyperparameters?
train_hyper = [True, True, True]

# For cross validation, do k splits with validation on val_frac of training data held out
cv = ShuffleSplit(n_splits=k, test_size=val_frac, random_state=seed)

# List of dicts of params for grid search cross validation
hyper_ranges = [
                {"C" : np.logspace(-5,2,n_c)},
                {"max_depth":[1,2,4,6,8,10,12,14,16,20]},
                {"C" : np.logspace(-5,2,n_c)}]

# Containers for error/loss metrics
train_r2 = []
train_loss = []
train_01_loss = []
test_r2 = []
test_loss = []
test_01_loss = []

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

        print(models[ii])

        # Save training R^2, MSE for train, test set

        # Train
        y_hat_train = models[ii].predict(X_train)
        train_loss.append(log_loss(y_train, y_hat_train))
        train_01_loss.append(zero_one_loss(y_train, y_hat_train))
        train_r2.append(models[ii].score(X_train, y_train))

        # Test
        y_hat_test = models[ii].predict(X_test)
        test_loss.append(log_loss(y_test, y_hat_test))
        test_01_loss.append(zero_one_loss(y_test, y_hat_test))
        test_r2.append(models[ii].score(X_test, y_test))


    # No hyperparameters, just fit on training data!
    else:
        print("Fitting:",models[ii])
        models[ii].fit(X_train, y_train)

        # Save training R^2, MSE for train, test set

        # Train
        y_hat_train = models[ii].predict(X_train)
        train_loss.append(log_loss(y_train, y_hat_train))
        train_01_loss.append(zero_one_loss(y_train, y_hat_train))
        train_r2.append(models[ii].score(X_train, y_train))

        # Test
        y_hat_test = models[ii].predict(X_test)
        test_loss.append(log_loss(y_test, y_hat_test))
        test_01_loss.append(zero_one_loss(y_test, y_hat_test))
        test_r2.append(models[ii].score(X_test, y_test))

print("Training, testing r^2:")
print(train_r2,test_r2)

print("Training, testing loss:")
print(train_loss,test_loss)

print("Training, testing 0/1 loss:")
print(train_01_loss,test_01_loss)

# Save models?
if save_models and not os.path.exists(os.path.join(cache_loc,phys_model_cache)):
    # Pickle the data to use with bootstrapping
    print("Caching data at %s" % os.path.join(cache_loc,phys_model_cache))
    with open(os.path.join(cache_loc,phys_model_cache), 'wb') as handle:
        pickle.dump(models, handle)

################################################################
#
# Perform bootstrapping using best models for non-ensemble
# estimators.
#
################################################################

# Specify bootstrapping parameters
nboots = 100

# Extract fitted linear models
lr = models[0]

# Bootstrap!
print("Bootstrapping...")
lr_mean, lr_std = bu.bootstrap_error_estimate_test(lr, X_train, y_train, X_test, nboots=nboots, seed=seed)

# Save to latex style table
data = [['LR',train_loss[0], test_loss[0], train_r2[0], test_r2[0], train_01_loss[0], test_01_loss[0], np.median(lr_std)],
        ['RF',train_loss[1], test_loss[1], train_r2[1], test_r2[1], train_01_loss[1], test_01_loss[1], 0.0],
        ['SVM',train_loss[2], test_loss[2], train_r2[2], test_r2[2],train_01_loss[2], test_01_loss[2],  0.0]]

col_names = ['Est','Train Loss','Test Loss', 'Train R2', 'Test R2', "Train 01 Loss", "Test 01 Loss", "Median Std"]

table = pd.DataFrame(data=data, columns=col_names)
print(table.to_latex())