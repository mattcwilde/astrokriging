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
import sys
sys.path.append("../Utils")
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle

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

# New Feature Cache locations for classification data
data_loc = "../../Data"
phys_cache = "prox_phys_class.pkl"
plot_loc = "../../Plots"

# Load physical data
if os.path.exists(os.path.join(data_loc,phys_cache)):
    print("Reading data from cache:",phys_cache)
    with open(os.path.join(data_loc,phys_cache), 'rb') as handle:
        X, y, names = pickle.load(handle)
else:
    raise NameError("%s not defined." % phys_cache)

fig, ax = plt.subplots()

x_name = "b_Semim"
y_name = "b_Ecce"
xind = names[x_name]
yind = names[y_name]

cax = ax.scatter(X[:,xind],X[:,yind], c=y, edgecolor="none", cmap="jet")
cbar = fig.colorbar(cax)
cbar.set_label("Critical Fraction",rotation=270,labelpad=20)

# Format
ax.set_xlim(X[:,xind].min(),X[:,xind].max())
ax.set_ylim(X[:,yind].min(),X[:,yind].max())
ax.set_xlabel("b Semimajor Axis [AU]")
ax.set_ylabel("b Eccentricity")

plt.show()

# Now, print grid of stuff where y == 1 to see if you can disentangle the priors
mask = y == 1
X_1 = X[mask]
X_0 = X[~mask]

print("1, 0 counts:",len(X_1),len(X_0))

# See what "posteriors" look like for c
"""
{u'Abs_Inc_Diff': 16, u'b_Ecce': 0, u'c_Mass': 21, u'b_TidalQ': 19, u'Abs_LongA_Diff': 17,
 u'b_Mass': 20, u'b_sinArgP': 5, u'b_DelH': 11, u'c_sinLongA': 8, u'b_DelG': 9,
 u'Abs_Ecce_Diff': 15, u'c_DelH': 12, u'c_Ecce': 3, u'Abs_ArgP_Diff': 18, u'b_Semim': 2,
 u'bc_DelH': 13, u'cb_DelH': 14, u'c_DelG': 10, u'c_sinArgP': 6, u'b_sinLongA': 7,
 u'c_Inc': 4, u'b_Inc': 1, u'star_Mass': 22}
"""

variables = ["c_Mass","c_Ecce","c_Inc"]

fig, axes = plt.subplots(nrows=3)

bins = 20
for ii, ax in enumerate(axes.flatten()):
    ax.hist(X_1[:,names[variables[ii]]], bins=bins, facecolor="blue", alpha=0.3)
    ax.hist(X_0[:,names[variables[ii]]], bins=bins, facecolor="red", alpha=0.3)    
    
    # Format
    ax.set_ylabel("Counts")    
    ax.set_xlabel(variables[ii].replace("_"," "))
    
fig.tight_layout()
plt.show()
