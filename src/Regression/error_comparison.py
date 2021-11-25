# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 10:53:16 2016

@author: dflemin3 [David P. Fleming, University of Washington, Seattle]

@email: dflemin3 (at) uw (dot) edu

This script compares/contrasts various methods for approximating the uncertainty of an 
estimator trained on VPLANET simulation results.  The gold-standard is in principle a
gaussian process, but other things like bootstrapping linear models

Here, I take the best GP as the ground truth and compute an 0/1 loss comparison between
the other GPs and bootstrapped models and evaluate it all on the testing set (for now?)

Results
-------

Truth = RQ

\begin{tabular}{llr}
\toprule
{} &  Estimator &   0/1Loss \\
\midrule
0 &        OLS &  0.155578 \\
1 &         RR &  0.155578 \\
2 &      GPRBF &  0.075038 \\
3 &  GPMattern &  0.030515 \\
4 &       GPRQ &  0.000000 \\
\bottomrule
\end{tabular}


Truth = Matern

\begin{tabular}{llr}
\toprule
{} &  Estimator &   0/1Loss \\
\midrule
0 &        OLS &  0.125063 \\
1 &         RR &  0.125063 \\
2 &      GPRBF &  0.044522 \\
3 &  GPMattern &  0.000000 \\
4 &       GPRQ &  0.030515 \\
\bottomrule
\end{tabular}


Truth = RBF

\begin{tabular}{llr}
\toprule
{} &  Estimator &   0/1Loss \\
\midrule
0 &        OLS &  0.080540 \\
1 &         RR &  0.080540 \\
2 &      GPRBF &  0.000000 \\
3 &  GPMattern &  0.044522 \\
4 &       GPRQ &  0.075038 \\
\bottomrule
\end{tabular}



"""

from __future__ import print_function, division, unicode_literals
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import bootstrap_utils as bu
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

#Typical plot parameters that make for pretty plots
mpl.rcParams['figure.figsize'] = (9,8)
mpl.rcParams['font.size'] = 20.0
mpl.rc('font',**{'family':'serif','serif':['Computer Modern']})
mpl.rc('text', usetex=True)

################################################################
#
# Load, preprocess data
#
################################################################

# Flags to control functionality
phys = True
poly = False # didn't train on this!
show_plots = False
save_plots = False

# Define constants
test_frac = 0.2
val_frac = 0.2 # Fraction of training data to use as validation for training hyperparams
k = 5 # number of folds for cross validation
seed = 42
nboots = 100

# Data, cache locations
data_loc = "../Data"
plot_loc = "../Plots"
phys_cache = "proc_physical_model.pkl"
data_cache = "proc_physical_3sig.pkl"
cache_loc = "/astro/store/gradscratch/tmp/dflemin3/ML_Data"
rbf_cache = "proc_phys_rbf_full.pkl"
matern_cache = "proc_phys_matern_full.pkl"
rq_cache = "proc_phys_rq_full.pkl"
truth_cache = rq_cache

# Load data
if os.path.exists(os.path.join(data_loc,data_cache)):
    print("Reading data from cache:",data_cache)
    with open(os.path.join(data_loc,data_cache), 'rb') as handle:
        X, y, names = pickle.load(handle)
else:
    raise NameError("%s not defined." % data_cache)

# Split data into training set, testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac,
                                                    random_state=seed)

# Scale data to 0 mean, 1 std based on training data
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

################################################################
#
# Examine std distributions on testing set
#
################################################################

# Load "truth" data
if os.path.exists(os.path.join(cache_loc,truth_cache)):
    print("Reading data from cache:",truth_cache)
    with open(os.path.join(cache_loc,truth_cache), 'rb') as handle:
        gp_rq = pickle.load(handle)
else:
    raise NameError("%s not defined." % truth_cache)
    
# See what the truth model is
print(gp_rq)    
    
# Predict!
print("Predicting on testing set for 'truth'...")
print("Truth: %s" % truth_cache)
print(gp_rq.kernel_)
test_y_hat, truth_std = gp_rq.predict(X_test, return_std=True)
gold_mask, median, gold_std = bu.find_outliers(truth_std)

# Visualize the "truth" distribution?
if show_plots:

    fig, ax = plt.subplots()
    
    ax.hist(truth_std, 20)
    ax.axvline(x=median, ls="--", color="red", lw=2, label="Median")
    ax.axvline(x=(median-gold_std), ls="--", lw=2, color="black")
    ax.axvline(x=(median+gold_std), ls="--", lw=2, color="black", label=r"1 $\sigma$")

    # Format
    ax.set_xlabel("Estimator Standard Deviation")
    ax.set_ylabel("Counts")

    ax.legend(loc="best")
    
    plt.show()
    
    if save_plots:
        fig.savefig(os.path.join(plot_loc,"rq_std_distribution.pdf"))
        
################################################################
#
# Compute 0/1 losses for other estimators
#
################################################################
        
models = ["linear","GPRBF","GPMattern","RQ"]
model_locs = ["proc_physical_model.pkl",
              "proc_phys_rbf_full.pkl",
              "proc_phys_matern_full.pkl",
              "proc_phys_rq_full.pkl"]

losses = []

# Loop over models, compute and store losses
for ii, model in enumerate(models):
    if model == "linear":
        
        # Extract fitted linear models
        print("Reading data from cache:",phys_cache)
        with open(os.path.join(cache_loc,phys_cache), 'rb') as handle:
            tmp = pickle.load(handle)
        ols = tmp[0]
        rr = tmp[1]
        
        # See what the models are like
        print(ols)
        print(rr)        
        
        # Bootstrap!
        print("Bootstrapping...")
        ols_mean, ols_std = bu.bootstrap_error_estimate_test(ols, X_train, y_train, X_test, nboots=nboots, seed=seed)
        rr_mean, rr_std = bu.bootstrap_error_estimate_test(rr, X_train, y_train, X_test, nboots=nboots, seed=seed)
        
        # Get loss on testing set
        print("Predicting...")
        test_mask = (y == y_test)
        ols_std = ols_std[test_mask]
        rr_std = rr_std[test_mask]
        ols_mask, median, std = bu.find_outliers(ols_std)
        rr_mask, median, std = bu.find_outliers(rr_std)                                
                
        print(bu.loss_01(gold_mask,ols_mask))
        print(bu.loss_01(gold_mask,rr_mask))        
        
        losses.append(bu.loss_01(gold_mask,ols_mask))
        losses.append(bu.loss_01(gold_mask,rr_mask))
        
    else:
        # Extract fitted linear models
        print("Reading data from cache:",model_locs[ii])
        with open(os.path.join(cache_loc,model_locs[ii]), 'rb') as handle:
            gp = pickle.load(handle)
            
        # See what the model is
        print(gp)            
            
        # Get loss on testing set
        print("Predicting...")
        test_y_hat, gp_std = gp.predict(X_test, return_std=True)
        gp_mask, median, gp_std = bu.find_outliers(gp_std)
        
        losses.append(bu.loss_01(gold_mask,gp_mask))
        
# To latex!
# Save to latex style table
data = [["OLS",losses[0]],
        ["RR",losses[1]],
        ["GPRBF",losses[2]],
        ["GPMattern",losses[3]],
        ["GPRQ",losses[4]]]

col_names = ["Estimator","0/1Loss"]

table = pd.DataFrame(data=data, columns=col_names)
print(table.to_latex())