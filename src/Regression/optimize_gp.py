# -*- coding: utf-8 -*-
"""
Created on Sunday Oct. 17th, 2016

@author: dflemin3 [David P Fleming, University of Washington]
@email: dflemin3 (at) uw (dot) edu

This file contains functions and routines for picking the optimal Gaussian process
for a given trianing set using cross-validation.

"""

from __future__ import print_function, division, unicode_literals

import numpy as np
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, RationalQuadratic
from bigplanet import data_extraction as de

if __name__ == "__main__":

    seed = 0
    n = 100
    show_plots = False

    ################################################################
    #
    # Load, process dataframe from cache
    #
    ################################################################

    print("Loading data...")
    data_loc = "../Data"
    df = de.aggregate_data(cache=os.path.join(data_loc,"Q100_cache.pkl"))

    # Get target label, data
    y = df["b_Prob_Frac"].values

    # Ignore all probability columns for features
    #cols = [col for col in df.columns if col not in ["b_Prob","b_Ecc_Prob","b_Prob_Frac",
    #                                                 "b_Semi_Prob",]]

    # Want features to be Prox c's parameters
    cols = ["c_Ecce","c_Inc"]
    X = df[cols].values
    X = X[:n,:]
    y = y[:n]

    # Define random seed
    seed = 0

    # Init Kernel as sum of ExpSinSquared kernel and WhiteKernel to estimate noise in data
    # Use default params as these will be optimized
    kernel = Matern() + WhiteKernel()

    # Init gaussian process
    print("Initializing Gaussian process...")
    n_restarts_optimizer=10
    gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=n_restarts_optimizer,
                                  random_state=seed,copy_X_train=False,optimizer='fmin_l_bfgs_b')

    print("Fitting GP...")
    gp.fit(X,y)
    print("Log Marginal Likelihood (optimized): %.3f"% gp.log_marginal_likelihood(gp.kernel_.theta))

    print(gp.kernel_)
    print("Score:",gp.score(X,y))