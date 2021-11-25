# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 14:21:47 2016

@author: dflemin3

This script contains various routines used for estimating whether or not Proxima b has 
existed near its observed orbital parameters within its estimated age range.  Effectively,
they return a probability/classification that says whether or not one or many properties
of Prox Cen b lies within a certain # of sigma of its observed values within some time
window.

Currently, these functions focus on the eccentricity and period since those are the direct
RV observables in addition to the semiamplitude.

These routines are designed to be used with bigplanet's aggregate_data or add_column
functions.

"""

# Imports
from __future__ import print_function, division
import proxima_params as pp
import numpy as np

def make_sigma(lower, upper, mean):
    """
    Given the lower and upper bound and the mean value, calculate the
    upper and lower sigmas (general for asymmetric error bars)  
    
    I think this works? TODO
    
    Parameters
    ----------
    lower : float
        1 sigma lower bound
    upper : float
        1 sigma upper bound
    mean : float
        mean (or median) value
    
    Returns
    -------
    msig, psig : floats
        lower 1 sigma, upper 1 sigma
    """
    msig = mean - lower
    psig = upper - mean
    
    return msig, psig
# end function


def ecc_target(data, i, body, fmt="hdf5", esigma=None, agemin=None,
               agemax=None, ecc_name=None,verbose=False,**kwargs):
    """
    Checks to see if at some point Prox b got within a certain number of sigma   
    of its observed ecc within specified age bounds.
    
    Parameters
    ----------
    data : Dataset Object
        bigplanet hdf5 dataset wrangler object
    i : int
        Simulation index
    body : string
        Name of the body whose parameters you want to mess with
    fmt : string (optional)
        For backwards compatibility.  Always hdf5.
    esigma : int (optional)
        How many standard deviations to look at (i.e. 3 is wider than 1)
    agemin : float (optional)
        minimum system age to consider (defaults to Barnes et al 2016)
    agemax : float (optional)
        maximum system age to consider (defaults to Barnes et al 2016)
    ecc_name : string (optional)
        Name of the column containing Prox b's eccentricity data
    verbose : bool (optional)
        If true, return the mask that shows when target was met

    Returns
    -------
    prob : float
        Whether or not the specified condition was satisfied at some point
    """
    
    # Defaults
    if esigma is None:
        esigma = 1
    if agemin is None:
        agemin = pp.AGE_ERR[0]
    if agemax is None:
        agemax = pp.AGE_ERR[1]
    if ecc_name is None:
        ecc_name = "Ecce"
    
    # Get the data of interest
    ecc, time = data.get(i,body,[ecc_name,"Time"])
    
    # Focus on plausible age range, mask data accordingly    
    time_mask = (time > agemin) & (time < agemax)
    ecc = ecc[time_mask]
    
    # Make sigma assuming a mean of 0 and scale by user-specified sigma
    msig, psig = make_sigma(pp.B_ECC_RANGE[0],pp.B_ECC_RANGE[1],0)  
    msig *= esigma
    psig *= esigma

    # Hard-code lower bound of 0, upper bound max of 1 from astrophysics    
    lower = 0
    upper = 0 + psig
    if upper > 1:
        upper = 1
        
    # For eccentricity, a sigma range doesn't make sense since it has an upper
    # and lower bound given by [0,0.35], so we'll default to 1 "sigma" here
    ecc_mask = (ecc > lower) & (ecc < upper)
        
    # If true at some point, return 1
    if np.sum(ecc_mask) > 0:
        pecc = 1
    else:
        pecc = 0
        
    if verbose:
        return pecc, ecc_mask
    else:
        return pecc
# end function
        
        
def semi_target(data, i, body, fmt="hdf5", ssigma=None, agemin=None,
               agemax=None, semi_name=None,verbose=False,**kwargs):
    """
    Checks to see if at some point Prox b got within a certain number of sigma   
    of its observed semimajox axis within specified age bounds.
    
    Parameters
    ----------
    data : Dataset Object
        bigplanet hdf5 dataset wrangler object
    i : int
        Simulation index
    body : string
        Name of the body whose parameters you want to mess with
    fmt : string (optional)
        For backwards compatibility.  Always hdf5.
    ssigma : int (optional)
        How many standard deviations to look at (3 is wider than 1)
    agemin : float (optional)
        minimum system age to consider (defaults to Barnes et al 2016)
    agemax : float (optional)
        maximum system age to consider (defaults to Barnes et al 2016)
    semi_name : string (optional)
        Name of the column containing Prox b's semimajor axis data
    verbose : bool (optional)
        If true, return the mask that shows when target was met

    Returns
    -------
    prob : float
        Whether or not the specified condition was satisfied at some point
    """
    
    # Defaults
    if ssigma is None:
        ssigma = 1
    if agemin is None:
        agemin = pp.AGE_ERR[0]
    if agemax is None:
        agemax = pp.AGE_ERR[1]
    if semi_name is None:
        semi_name = "Semim"
    
    # Get the data of interest
    semi, time = data.get(i,body,[semi_name,"Time"])
    
    # Focus on plausible age range, mask data accordingly    
    time_mask = (time > agemin) & (time < agemax)
    semi = semi[time_mask]
    
     # Make sigma assuming a mean of 0 and scale by user-specified sigma
    msig, psig = make_sigma(pp.B_SEMI_ERR[0],pp.B_SEMI_ERR[1],pp.B_SEMI)
        
    # Scale sigmas, enfore astrophysical bounds
    msig *= ssigma
    psig *= ssigma
    
    lower = pp.B_SEMI - msig
    if lower <= 0:
        lower = 0.03 # Probs can't be lower than this
    upper = pp.B_SEMI + psig
    
    semi_mask = (semi > lower) & (semi < upper)
    
    # If true at some point, return 1
    if np.sum(semi_mask) > 0:
        psemi = 1
    else:
        psemi = 0
        
    if verbose:
        return psemi, semi_mask
    else:
        return psemi
# end function
        
        
def period_target(data, i, body, fmt="hdf5", psigma=None, agemin=None,
               agemax=None, period_name=None,verbose=False,**kwargs):
    """
    Checks to see if at some point Prox b got within a certain number of sigma   
    of its observed period within specified age bounds.
    
    Parameters
    ----------
    data : Dataset Object
        bigplanet hdf5 dataset wrangler object
    i : int
        Simulation index
    body : string
        Name of the body whose parameters you want to mess with
    fmt : string (optional)
        For backwards compatibility.  Always hdf5.
    psigma : int (optional)
        How many standard deviations to look at (3 is wider than 1)
    agemin : float (optional)
        minimum system age to consider (defaults to Barnes et al 2016)
    agemax : float (optional)
        maximum system age to consider (defaults to Barnes et al 2016)
    period_name : string (optional)
        Name of the column containing Prox b's period data
    verbose : bool (optional)
        If true, return the mask that shows when target was met

    Returns
    -------
    prob : float
        Whether or not the specified condition was satisfied at some point
    """
    
    # Defaults
    if psigma is None:
        psigma = 1
    if agemin is None:
        agemin = pp.AGE_ERR[0]
    if agemax is None:
        agemax = pp.AGE_ERR[1]
    if period_name is None:
        period_name = "Period"
    
    # Get the data of interest
    period, time = data.get(i,body,[period_name,"Time"])
    
    # Focus on plausible age range, mask data accordingly    
    time_mask = (time > agemin) & (time < agemax)
    period = period[time_mask]
    
     # Make sigma assuming a mean of 0 and scale by user-specified sigma
    msig, psig = make_sigma(pp.B_PERIOD_ERR[0],pp.B_PERIOD_ERR[1],pp.B_PERIOD)
        
    # Scale sigmas, enfore astrophysical bounds
    msig *= psigma
    psig *= psigma
    
    lower = pp.B_PERIOD - msig
    upper = pp.B_PERIOD + psig
    
    period_mask = (period > lower) & (period < upper)
    
    # If true at some point, return 1
    if np.sum(period_mask) > 0:
        pperiod = 1
    else:
        pperiod = 0
        
    if verbose:
        return pperiod, period_mask
    else:
        return pperiod
# end function
        

def semi_ecc_target(data, i, body, fmt="hdf5", ssigma=None, agemin=None,
                    agemax=None, semi_name=None, esigma=None, ecc_name=None,
                    **kwargs):
    """
    Checks to see if at some point Prox b got within a certain number of sigma   
    of its observed semimajox axis and eccentricity within specified age bounds.
    
    Parameters
    ----------
    Same as semi_target and ecc_target

    Returns
    -------
    prob : float
        Whether or not the specified condition was satisfied at some point    
    """
    
    # Defaults
    if ssigma is None:
        ssigma = 1
    if agemin is None:
        agemin = pp.AGE_ERR[0]
    if agemax is None:
        agemax = pp.AGE_ERR[1]
    if semi_name is None:
        semi_name = "Semim"
    if esigma is None:
        esigma = 1
    if ecc_name is None:
        ecc_name = "Ecce"
    
    # Get semimajor axis probability
    psemi, semi_mask = semi_target(data, i, body, fmt="hdf5", ssigma=ssigma, 
                                   agemin=agemin,agemax=agemax,
                                   semi_name=semi_name,verbose=True,**kwargs)
                                   
    # Get eccentricity probability
    pecc, ecc_mask = ecc_target(data, i, body, fmt="hdf5", esigma=esigma, agemin=agemin,
                                agemax=agemax, ecc_name=ecc_name,verbose=True,**kwargs)
    
                   
    # Compute whether or not the conditions occur simultaneously
    ptot = np.sum(np.logical_and(ecc_mask,semi_mask))
    if ptot >= 1:
        ptot = 1
    else:
        ptot = 0
                   
    return ptot
# end function
    
    
def period_ecc_target(data, i, body, fmt="hdf5", psigma=None, agemin=None,
                    agemax=None, period_name=None, esigma=None, ecc_name=None,
                    **kwargs):
    """
    Checks to see if at some point Prox b got within a certain number of sigma   
    of its observed period and eccentricity within specified age bounds.
    
    Parameters
    ----------
    Same as period_target and ecc_target

    Returns
    -------
    prob : float
        Whether or not the specified condition was satisfied at some point    
    """
    
    # Defaults
    if psigma is None:
        psigma = 1
    if agemin is None:
        agemin = pp.AGE_ERR[0]
    if agemax is None:
        agemax = pp.AGE_ERR[1]
    if period_name is None:
        period_name = "Period"
    if esigma is None:
        esigma = 1
    if ecc_name is None:
        ecc_name = "Ecce"
    
    # Get semimajor axis probability
    pperiod, period_mask = period_target(data, i, body, fmt="hdf5", psigma=psigma, 
                                   agemin=agemin,agemax=agemax,
                                   period_name=period_name,verbose=True,**kwargs)
                                   
    # Get eccentricity probability
    pecc, ecc_mask = ecc_target(data, i, body, fmt="hdf5", esigma=esigma, agemin=agemin,
                                agemax=agemax, ecc_name=ecc_name,verbose=True,**kwargs)
    
                   
    # Compute whether or not the conditions occur simultaneously
    ptot = np.sum(np.logical_and(ecc_mask,period_mask))
    if ptot >= 1:
        ptot = 1
    else:
        ptot = 0
                   
    return ptot
# end function
    
    
def period_ecc_frac(data, i, body, fmt="hdf5", psigma=None, agemin=None,
                  agemax=None, period_name=None, esigma=None, ecc_name=None,
                  **kwargs):
    """
    Checks to see what fraction of the total simulation time point Prox b got within a 
    certain number of sigma of its observed period and eccentricity within 
    specified age bounds.
    
    Parameters
    ----------
    Same as period_target and ecc_target

    Returns
    -------
    prob : float
        Whether or not the specified condition was satisfied at some point    
    """
    
    # Defaults
    if psigma is None:
        psigma = 1
    if agemin is None:
        agemin = pp.AGE_ERR[0]
    if agemax is None:
        agemax = pp.AGE_ERR[1]
    if period_name is None:
        period_name = "Period"
    if esigma is None:
        esigma = 1
    if ecc_name is None:
        ecc_name = "Ecce"    
        
    # Get semimajor axis probability
    pperiod, period_mask = period_target(data, i, body, fmt="hdf5", psigma=psigma, 
                                   agemin=agemin,agemax=agemax,
                                   period_name=period_name,verbose=True,**kwargs)
                                   
    # Get eccentricity probability
    pecc, ecc_mask = ecc_target(data, i, body, fmt="hdf5", esigma=esigma, agemin=agemin,
                                agemax=agemax, ecc_name=ecc_name,verbose=True, **kwargs)
    
                   
    # Compute fraction of time in which conditions occur simultaneously
    return np.sum(np.logical_and(ecc_mask,period_mask))/len(ecc_mask)
# end function
    
    
def semi_ecc_frac(data, i, body, fmt="hdf5", ssigma=None, agemin=None,
                  agemax=None, semi_name=None, esigma=None, ecc_name=None,
                  **kwargs):
    """
    Checks to see what fraction of the total simulation time point Prox b got within a 
    certain number of sigma of its observed semimajor axis and eccentricity within 
    specified age bounds.
    
    Parameters
    ----------
    Same as semi_target and ecc_target

    Returns
    -------
    prob : float
        Whether or not the specified condition was satisfied at some point    
    """
    
    # Defaults
    if ssigma is None:
        ssigma = 1
    if agemin is None:
        agemin = pp.AGE_ERR[0]
    if agemax is None:
        agemax = pp.AGE_ERR[1]
    if semi_name is None:
        semi_name = "Semim"
    if esigma is None:
        esigma = 1
    if ecc_name is None:
        ecc_name = "Ecce"    
        
    # Get semimajor axis probability
    psemi, semi_mask = semi_target(data, i, body, fmt="hdf5", ssigma=ssigma, 
                                   agemin=agemin,agemax=agemax,
                                   semi_name=semi_name,verbose=True,**kwargs)
                                   
    # Get eccentricity probability
    pecc, ecc_mask = ecc_target(data, i, body, fmt="hdf5", esigma=esigma, agemin=agemin,
                                agemax=agemax, ecc_name=ecc_name,verbose=True, **kwargs)
    
                   
    # Compute fraction of time in which conditions occur simultaneously
    return np.sum(np.logical_and(ecc_mask,semi_mask))/len(ecc_mask)
# end function
    
    
####################################################################################
#
# Functions that cook up new features for simulations
#
# None currently implemented...
#
####################################################################################
    

def param_max(data, i, body, param="Ecce", fmt="hdf5", **kwargs):
    """
    Checks to see if at some point Prox b got within a certain number of sigma   
    of its observed semimajox axis within specified age bounds.
    
    Parameters
    ----------
    data : Dataset Object
        bigplanet hdf5 dataset wrangler object
    i : int
        Simulation index
    body : string
        Name of the body whose parameters you want to mess with
    param : string
        Name of parameter to compute maximum 
    fmt : string (optional)
        For backwards compatibility.  Always hdf5.

    Returns
    -------
    max : float
        Maximum value of given param over simulation
    """
    
    # Get the data of interest
    semi, time = data.get(i,body,[param,"Time"])
    return None