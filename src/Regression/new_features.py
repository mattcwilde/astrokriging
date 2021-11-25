# -*- coding: utf-8 -*-
"""
Created on Wed Nov 2 2016

@author: dflemin3

This script explores what happens when I cook up new features that are a function of
initial conditions to see if the fit improves.  Effectively, this script makes new features
and caches the results if it hasn't been ran already.

Features
--------

"""
# Imports
from __future__ import print_function, division, unicode_literals
import os
import numpy as np
import pandas as pd
import pickle

# bigplanet Imports
from bigplanet import data_extraction as de
from bigplanet import big_ml as bml

# Flags to control functionality
make_new_features = True
sklearn_fit = False
verbose = False

################################################################
#
# Load dataframe from cache, exploratory data analysis
#
################################################################

# Cache locations
data_loc = "../Data"
phys_cache = "proc_physical_3sig.pkl"
phys_poly_cache = "proc_physical_poly_3sig.pkl"
fourier_cache = "proc_fourier_3sig.pkl"
nn_cache = "proc_nn_3sig.pkl"

# Parameters
train_frac = 0.8 # Fraction of data to consider "training set"

# Open the original vplanet sim result dataframe
df = de.aggregate_data(cache=os.path.join(data_loc,"proc_sims_3sig.pkl"))

# Cook up new features?
if make_new_features:

    # If you haven't already...
    if os.path.exists(os.path.join(data_loc,phys_cache)):
        print("Reading data from cache:",os.path.join(data_loc,phys_cache))
        with open(os.path.join(data_loc,phys_cache), 'rb') as handle:
            X_phys, y_phys, names = pickle.load(handle)
    else:
        # Transform ArgP, LongA to sin of those quantities to handle edge cases
        # for all bodies
        df["b_sinArgP"] = pd.Series(np.sin(df["b_ArgP"]), index=df.index)
        df["c_sinArgP"] = pd.Series(np.sin(df["c_ArgP"]), index=df.index)
        df["b_sinLongA"] = pd.Series(np.sin(df["b_LongA"]), index=df.index)
        df["c_sinLongA"] = pd.Series(np.sin(df["c_LongA"]), index=df.index)

        # Add Delauney variables DelG, DelH defined as follows:
        # DelG ~ sqrt(1 - e^2)
        # DelH ~ sqrt(1 - e^2)cosi
        # Note I drop other constants as they just scale it
        df["b_DelG"] = pd.Series(np.sqrt(1.0 - df["b_Ecce"]**2), index=df.index)
        df["c_DelG"] = pd.Series(np.sqrt(1.0 - df["c_Ecce"]**2), index=df.index)
        df["b_DelH"] = pd.Series(np.cos(df["b_Inc"])*np.sqrt(1.0 - df["b_Ecce"]**2), index=df.index)
        df["c_DelH"] = pd.Series(np.cos(df["c_Inc"])*np.sqrt(1.0 - df["c_Ecce"]**2), index=df.index)

        # Add mixed Delauney variables like sqrt(1 - e_b^2)cos(i_c)
        df["bc_DelH"] = pd.Series(np.cos(df["b_Inc"])*np.sqrt(1.0 - df["c_Ecce"]**2), index=df.index)
        df["cb_DelH"] = pd.Series(np.cos(df["c_Inc"])*np.sqrt(1.0 - df["b_Ecce"]**2), index=df.index)

        # Add differences of variables, like abs diff between angles, ecc, inc, so on
        df["Abs_Ecce_Diff"] = pd.Series(np.fabs(df["b_Ecce"] - df["c_Ecce"]), index=df.index)
        df["Abs_Inc_Diff"] = pd.Series(np.fabs(df["b_Inc"] - df["c_Inc"]), index=df.index)
        df["Abs_LongA_Diff"] = pd.Series(np.fabs(df["b_LongA"] - df["c_LongA"]), index=df.index)
        df["Abs_ArgP_Diff"] = pd.Series(np.fabs(df["b_ArgP"] - df["c_ArgP"]), index=df.index)

        # Create data, filter NaNs
        features = ['b_Ecce', 'b_Inc','b_Semim','c_Ecce','c_Inc','b_sinArgP',
                'c_sinArgP', 'b_sinLongA', 'c_sinLongA', 'b_DelG', 'c_DelG','b_DelH','c_DelH',
                'bc_DelH','cb_DelH','Abs_Ecce_Diff','Abs_Inc_Diff','Abs_LongA_Diff',
                'Abs_ArgP_Diff','b_TidalQ','b_Mass','c_Mass','star_Mass']
        target = "b_Prob_Frac"
        X_phys, y_phys, names = bml.extract_features(df, features, target)
        res = [X_phys, y_phys, names]

        # Now cache...
        print("Caching data at %s" % os.path.join(data_loc,phys_cache))
        with open(os.path.join(data_loc,phys_cache), 'wb') as handle:
            pickle.dump(res, handle)

    # Ok, now make some polynomial expansion of these augmented physical features if you
    # haven't already
    if os.path.exists(os.path.join(data_loc,phys_poly_cache)):
        print("Reading data from cache:",os.path.join(data_loc,phys_poly_cache))
        with open(os.path.join(data_loc,phys_poly_cache), 'rb') as handle:
            X_poly, y_poly, names = pickle.load(handle)
    # Don't have it so blow up physical feature space
    else:
        X_poly = bml.poly_features(X_phys, degree=2)

        # Names is meaningless here because of the polynomial transformation,
        # but save it anyways
        names = {}
        res = [X_poly, y_phys, names]

        # Now cache...
        print("Caching data at %s" % os.path.join(data_loc,phys_poly_cache))
        with open(os.path.join(data_loc,phys_poly_cache), 'wb') as handle:
            pickle.dump(res, handle)

    # Now blow up physical space with a naive neural network transformation
    if os.path.exists(os.path.join(data_loc,nn_cache)):
        print("Reading data from cache:",os.path.join(data_loc,nn_cache))
        with open(os.path.join(data_loc,nn_cache), 'rb') as handle:
            X_nn, y_nn, names = pickle.load(handle)
    # Generate em!
    else:
        # Pick k to be length of "training set"
        k = int(X_phys.shape[0]*train_frac)
        X_nn, v_nn = bml.naive_nn_layer(X_phys, k=k, verbose=True)

        # Save stuff including transformation matrix v_nn!
        names = {}
        res = [X_nn, y_phys, names, v_nn]

        # Don't cache this since the matrix is like 2 gigs
        """
        print("Caching data at %s" % os.path.join(data_loc,nn_cache))
        with open(os.path.join(data_loc,nn_cache), 'wb') as handle:
            pickle.dump(res, handle)
        """


################################################################
#
# Test some of the new feature spaces
#
################################################################

if sklearn_fit:

    # Extra imports for testing
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import bootstrap_utils as bu

    # Load poly data
    if os.path.exists(os.path.join(data_loc,phys_poly_cache)):
        print("Reading data from cache:",phys_poly_cache)
        with open(os.path.join(data_loc,phys_poly_cache), 'rb') as handle:
            Xpoly, y, names = pickle.load(handle)

    # Load physical data
    if os.path.exists(os.path.join(data_loc,phys_cache)):
        print("Reading data from cache:",phys_cache)
        with open(os.path.join(data_loc,phys_cache), 'rb') as handle:
            X, y, names = pickle.load(handle)

    print(Xpoly.shape)

    # Import models
    from sklearn.linear_model import LinearRegression

    # Create data, filter NaNs
    features = ['b_Ecce', 'b_Inc','b_Semim','c_Ecce','c_Inc','b_sinArgP',
            'c_sinArgP', 'b_sinLongA', 'c_sinLongA', 'b_DelG', 'c_DelG','b_DelH','c_DelH',
            'bc_DelH','cb_DelH','Abs_Ecce_Diff','Abs_Inc_Diff','Abs_LongA_Diff','Abs_ArgP_Diff']
    target = "b_Prob_Frac"
    # turns them into arrays
    #X, y, names = bml.extract_features(df, features, target)

    # Scale data to have 0 mean, unit standard deviation
    X_scaled = bml.scale_data(Xpoly)

    # Bootstrapping Parameters
    nboots = 50
    seed = 1

    # Use simple linear model
    est = LinearRegression()

    print("Fitting...")
    fit_mean, fit_std = bu.bootstrap_error_estimate(est, X_scaled, y, nboots=nboots, seed=seed)
    print(est.score(X_scaled,y))

    # Project into 2d space
    fig, ax = plt.subplots()

    xind = names["b_Inc"]
    yind = names["c_Inc"]

    cax = ax.scatter(X[:,xind],X[:,yind], c=fit_std, edgecolor="none")
    cbar = fig.colorbar(cax)

    # Format
    ax.set_xlim(X[:,xind].min(),X[:,xind].max())
    ax.set_ylim(X[:,yind].min(),X[:,yind].max())
    ax.set_xlabel("b_Inc")
    ax.set_ylabel("c_Inc")

    plt.show()
