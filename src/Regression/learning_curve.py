# Imports
from __future__ import print_function, division, unicode_literals
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing
from sklearn.model_selection import learning_curve

#Typical plot parameters that make for pretty plots
mpl.rcParams['figure.figsize'] = (9,8)
mpl.rcParams['font.size'] = 20.0
mpl.rc('font',**{'family':'serif','serif':['Computer Modern']})
mpl.rc('text', usetex=True)

# copy sklearn learning curve function for easy use/plotting
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Flags to control functionality
save_plots = True

# Constants
seed = 42 # RNG seed
test_frac = 0.2
k = 25 # number of folds for cross validation

# Locations of caches, data
data_loc = "../Data"
phys_cache = "proc_physical_3sig.pkl"
phys_poly_cache = "proc_physical_poly_3sig.pkl"
plot_loc = "../Plots"

################################################################
#
# Load dataframe from cache, exploratory data analysis
#
################################################################

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
# PHYS: Learning curve
#
################################################################

# Split into training, testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_frac,
                                                    random_state=seed)

# Scale data to 0 mean, 1 std based on training data
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Learning curve!
cv = ShuffleSplit(n_splits=k, test_size=test_frac, random_state=seed)

est = LinearRegression()
title = "Physical Features Linear Regression Learning Curve"
plot = plot_learning_curve(est, title, X_train, y_train, cv=cv)
plot.show()

if save_plots:
    plot.savefig("phys_lr_learning_curve.pdf")

################################################################
#
# POLY: Learning curve
#
################################################################

# Split into training, testing set
X_train, X_test, y_train, y_test = train_test_split(Xpoly, y,
                                                    test_size = test_frac,
                                                    random_state=seed)

# Scale data to 0 mean, 1 std based on training data
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Learning curve!
cv = ShuffleSplit(n_splits=k, test_size=test_frac, random_state=seed)

est = LinearRegression()
title = "Polynomial Features Linear Regression Learning Curve"
plot = plot_learning_curve(est, title, X_train, y_train, ylim=(0.5,0.8), cv=cv)
plot.show()

if save_plots:
    plot.savefig("poly_lr_learning_curve.pdf")
