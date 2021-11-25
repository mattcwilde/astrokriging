# -*- coding: utf-8 -*-
"""
Created on Sat. Nov 26, 2016

@author: dflemin3 [David P Fleming, University of Washington]
@email: dflemin3 (at) uw (dot) edu

Fit GP models with various kernels to data, cache results.  Here, we fit just on the
physical dataset.  Other datasets likely have too large of a dimensionality for a fit
to finish in a reasonable time (this is only like 10000 x 20)

\begin{tabular}{llrrrr}
\toprule
{} &           GPKernel &  Training MSE &  Testing MSE &  Training R2 &  Testing R2 \\
\midrule
0 &             Matern &      0.050634 &     0.081951 &     0.788365 &    0.656356 \\
1 &                RBF &      0.065208 &     0.083292 &     0.727450 &    0.650734 \\
2 &  RationalQuadratic &      0.046516 &     0.081516 &     0.805579 &    0.658183 \\
\bottomrule
\end{tabular}


"""

from __future__ import print_function, division, unicode_literals

import os
import pickle
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, RationalQuadratic
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import pandas as pd
import numpy as np

# Parameters, Flags to control functionality
seed = 42
n_opts = 10
test_frac = 0.2
fit_RBF = True
fit_Matern = True
fit_RQ = True
save_fit = True

# Cache locations
data_loc = "../Data"
# cache_loc = "/astro/store/gradscratch/tmp/dflemin3/ML_Data"
cache_loc = "../../../../../../Data"
phys_cache = "proc_physical_3sig.pkl"
plot_loc = "../Plots"
rbf_cache = "proc_phys_rbf_full.pkl"
matern_cache = "proc_phys_matern_full.pkl"
rq_cache = "proc_phys_rq_full.pkl"

# Error holders
train_mse = []
test_mse = []
train_r2 = []
test_r2 = []
<<<<<<< HEAD
train_std = []
test_std = []
=======
std = []
>>>>>>> e73802a6a07344dfcc941ceecc1cb37d054efba4

################################################################
#
# Load, process dataframe from cache
#
################################################################

# Load physical feature data
if os.path.exists(os.path.join(data_loc,phys_cache)):
    print("Reading data from cache:",phys_cache)
    with open(os.path.join(data_loc,phys_cache), 'rb') as handle:
        X, y, names = pickle.load(handle)
else:
    raise NameError("%s not defined." % phys_cache)

# Split into training, testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_frac,
                                                    random_state=seed)

# Scale data to 0 mean, 1 std based on training data
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

################################################################
#
# Fit with a whole mess of kernels (if I haven't already)
# Otherwise, load fit and get errors/scores
#
################################################################

# Try Matern kernel?
if fit_Matern:
    if not os.path.exists(os.path.join(cache_loc,matern_cache)):
        print("Fitting with White + Matern kernel...")
        kernel = WhiteKernel() + Matern()

        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_opts,
                                      alpha = 0.0,
                                      random_state=seed)
        print("Fitting...")
        gp.fit(X_train,y_train)
    else:
        print("Reading data from cache:",matern_cache)
        with open(os.path.join(cache_loc,matern_cache), 'rb') as handle:
            gp = pickle.load(handle)

    # Train
    y_hat_train, ytrain_std = gp.predict(X_train, return_std=True)
    train_mse.append(mean_squared_error(y_train, y_hat_train))
    train_r2.append(gp.score(X_train, y_train))
    train_std.append(np.median(ytrain_std))

    # Test
<<<<<<< HEAD
    y_hat_test, ytest_std = gp.predict(X_test, return_std=True)
=======
    y_hat_test, test_std = gp.predict(X_test, return_std=True)
    std.append(np.mean(test_std))
>>>>>>> e73802a6a07344dfcc941ceecc1cb37d054efba4
    test_mse.append(mean_squared_error(y_test, y_hat_test))
    test_r2.append(gp.score(X_test, y_test))
    test_std.append(np.median(ytest_std))

    # Output to the screen
    print("Train, test scores:")
    print(gp.score(X_train,y_train))
    print(gp.score(X_test,y_test))

    # Now cache...
    if save_fit and not os.path.exists(os.path.join(cache_loc,matern_cache)):
        print("Caching data at %s" % os.path.join(cache_loc,matern_cache))
        with open(os.path.join(cache_loc,matern_cache), 'wb') as handle:
            pickle.dump(gp, handle)

# Try RBF kernel?
if fit_RBF:
    if not os.path.exists(os.path.join(cache_loc,rbf_cache)):
        print("Fitting with White + RBF kernel...")
        kernel = WhiteKernel() + RBF()

        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_opts,
                                      alpha = 0.0,
                                      random_state=seed)
        print("Fitting...")
        gp.fit(X_train,y_train)
    else:
        print("Reading data from cache:",rbf_cache)
        with open(os.path.join(cache_loc,rbf_cache), 'rb') as handle:
            gp = pickle.load(handle)

    # Train
    y_hat_train, ytrain_std = gp.predict(X_train, return_std=True)
    train_mse.append(mean_squared_error(y_train, y_hat_train))
    train_r2.append(gp.score(X_train, y_train))
    train_std.append(np.median(ytrain_std))

<<<<<<< HEAD
    # Test
    y_hat_test, ytest_std = gp.predict(X_test, return_std=True)
=======
   # Test
    y_hat_test, test_std = gp.predict(X_test, return_std=True)
    std.append(np.mean(test_std))
>>>>>>> e73802a6a07344dfcc941ceecc1cb37d054efba4
    test_mse.append(mean_squared_error(y_test, y_hat_test))
    test_r2.append(gp.score(X_test, y_test))
    test_std.append(np.median(ytest_std))

    # Now cache...
    if save_fit and not os.path.exists(os.path.join(cache_loc,rbf_cache)):
        print("Caching data at %s" % os.path.join(cache_loc,rbf_cache))
        with open(os.path.join(cache_loc,rbf_cache), 'wb') as handle:
            pickle.dump(gp, handle)

if fit_RQ:
    if not os.path.exists(os.path.join(cache_loc,rq_cache)):
        print("Fitting with White + Rational Quadratic kernel...")
        kernel = WhiteKernel() + RationalQuadratic()

        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_opts,
                                      alpha = 0.0,
                                      random_state=seed)
        print("Fitting...")
        gp.fit(X_train,y_train)
    else:
        print("Reading data from cache:",rq_cache)
        with open(os.path.join(cache_loc,rq_cache), 'rb') as handle:
            gp = pickle.load(handle)

    # Train
    y_hat_train, ytrain_std = gp.predict(X_train, return_std=True)
    train_mse.append(mean_squared_error(y_train, y_hat_train))
    train_r2.append(gp.score(X_train, y_train))
    train_std.append(np.median(ytrain_std))

    # Test
<<<<<<< HEAD
    y_hat_test, ytest_std = gp.predict(X_test, return_std=True)
=======
    y_hat_test, test_std = gp.predict(X_test, return_std=True)
    std.append(np.mean(test_std))
>>>>>>> e73802a6a07344dfcc941ceecc1cb37d054efba4
    test_mse.append(mean_squared_error(y_test, y_hat_test))
    test_r2.append(gp.score(X_test, y_test))
    test_std.append(np.median(ytest_std))

    # Now cache...
    if save_fit and not os.path.exists(os.path.join(cache_loc,rq_cache)):
        print("Caching data at %s" % os.path.join(cache_loc,rq_cache))
        with open(os.path.join(cache_loc,rq_cache), 'wb') as handle:
            pickle.dump(gp, handle)


################################################################
#
# Make data pretty again
#
################################################################

# Save to latex style table
<<<<<<< HEAD
data = [['Matern',train_mse[0], test_mse[0], train_r2[0], test_r2[0], train_std[0], test_std[0]],
        ['RBF',train_mse[1], test_mse[1], train_r2[1], test_r2[1], train_std[1], test_std[1]],
        ['RationalQuadratic',train_mse[2], test_mse[2], train_r2[2], test_r2[2], train_std[2], test_std[2]]]

col_names = ['GPKernel','Training MSE','Testing MSE', 'Training R2', 'Testing R2', 'Training median std', 'Testing median std']
=======
data = [['Matern',train_mse[0], test_mse[0], train_r2[0], test_r2[0], std[0]],
        ['RBF',train_mse[1], test_mse[1], train_r2[1], test_r2[1], std[1]],
        ['RationalQuadratic',train_mse[2], test_mse[2], train_r2[2], test_r2[2], std[2]]]

col_names = ['GPKernel','Training MSE','Testing MSE', 'Training R2', 'Testing R2', 'Mean Std']
>>>>>>> e73802a6a07344dfcc941ceecc1cb37d054efba4

table = pd.DataFrame(data=data, columns=col_names)
print(table.to_latex())

# Done!
