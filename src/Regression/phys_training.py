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

\begin{tabular}{llrrrrr}
\toprule
{} &  est &  training MSE &  testing MSE &  training R\textasciicircum2 &  testing R\textasciicircum2 &  Median Std \\
\midrule
0 &  OLS &      0.099971 &     0.102640 &      0.582150 &     0.569601 &    0.015973 \\
1 &   RR &      0.099974 &     0.102606 &      0.582140 &     0.569746 &    0.015426 \\
2 &   RF &      0.014023 &     0.033080 &      0.941388 &     0.861286 &    0.000000 \\
\bottomrule
\end{tabular}



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
from sklearn import preprocessing

#Typical plot parameters that make for pretty plots
mpl.rcParams['figure.figsize'] = (9,8)
mpl.rcParams['font.size'] = 20.0
mpl.rc('font',**{'family':'serif','serif':['Computer Modern']})
mpl.rc('text', usetex=True)


# Flags to control functionality
show_plots = False
save_models = True

# Constants
seed = 42 # RNG seed
test_frac = 0.2
val_frac = 0.2 # Fraction of training data to use as validation for training hyperparams
n_alpha = 50 # Size of alpha grid search for ridge
k = 5 # number of folds for cross validation

# Locations of caches, data
data_loc = "../Data"
phys_cache = "proc_physical_3sig.pkl"
phys_poly_cache = "proc_physical_poly_3sig.pkl"
phys_model_cache = "proc_physical_model.pkl"
cache_loc = "/astro/store/gradscratch/tmp/dflemin3/ML_Data"
plot_loc = "../Plots"

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

        print(models[ii])

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
rr = models[1]

# Bootstrap!
print("Bootstrapping...")
ols_mean, ols_std = bu.bootstrap_error_estimate_test(lr, X_train, y_train, X_test, nboots=nboots, seed=seed)
rr_mean, rr_std = bu.bootstrap_error_estimate_test(rr, X_train, y_train, X_test, nboots=nboots, seed=seed)

# Save to latex style table
data = [['OLS',train_mse[0], test_mse[0], train_r2[0], test_r2[0], np.median(ols_std)],
        ['RR',train_mse[1], test_mse[1], train_r2[1], test_r2[1], np.median(rr_std)],
        ['RF',train_mse[2], test_mse[2], train_r2[2], test_r2[2], 0.0]]

col_names = ['est','training MSE','testing MSE', r'training R^2', r'testing R^2', r"Median Std"]

table = pd.DataFrame(data=data, columns=col_names)
print(table.to_latex())

################################################################
#
# Visualize bootstrapping results
#
################################################################

if show_plots:

    # See how ridge regression bootstrapping performs
    fig, ax = plt.subplots()

    xind = names["b_Inc"]
    yind = names["c_Inc"]

    cax = ax.scatter(X[:,xind],X[:,yind], c=rr_std, edgecolor="none", cmap="viridis")
    cbar = fig.colorbar(cax)
    cbar.set_label("Standard Deviation",rotation=270,labelpad=20)

    # Format
    ax.set_xlim(X[:,xind].min(),X[:,xind].max())
    ax.set_ylim(X[:,yind].min(),X[:,yind].max())
    ax.set_xlabel("b Inclination [degrees]")
    ax.set_ylabel("c Inclination [degrees]")

    #fig.tight_layout()
    #fig.savefig(os.path.join(plot_loc,"rr_inc_inc.pdf"))

    plt.show()
