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
import pickle
import proxima_params as pp

#Typical plot parameters that make for pretty plots
mpl.rcParams['figure.figsize'] = (9,8)
mpl.rcParams['font.size'] = 20.0
mpl.rc('font',**{'family':'serif','serif':['Computer Modern']})
mpl.rc('text', usetex=True)

# Flags to control functionality
seed = 42 # RNG seed

################################################################
#
# Load dataframe from cache, exploratory data analysis
#
################################################################

data_loc = "../Data"
phys_cache = "df_physical_3sig.pkl"
plot_loc = "../Plots"

# Load physical data
if os.path.exists(os.path.join(data_loc,phys_cache)):
    print("Reading data from cache:",phys_cache)
    with open(os.path.join(data_loc,phys_cache), 'rb') as handle:
        X, y, names = pickle.load(handle)
else:
    raise NameError("%s not defined." % phys_cache)

fig, ax = plt.subplots()

ax.hist(X[:,names["b_Semim"]], bins=20)
true = pp.B_SEMI
ax.axvline(x=true, ls="--", lw=2, color="black")
plt.show()

fig, ax = plt.subplots()

x_name = "b_Semim"
y_name = "b_Ecce"
xind = names[x_name]
yind = names[y_name]

cax = ax.scatter(X[:,xind],X[:,yind], c=y, edgecolor="none", cmap="viridis")
cbar = fig.colorbar(cax)
cbar.set_label("Critical Fraction",rotation=270,labelpad=20)

# Format
ax.set_xlim(X[:,xind].min(),X[:,xind].max())
ax.set_ylim(X[:,yind].min(),X[:,yind].max())
ax.set_xlabel("b Semimajor Axis [AU]")
ax.set_ylabel("b Eccentricity")

plt.show()
