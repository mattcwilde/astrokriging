# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 15:25:26 2016

@author: dflemin3

This helper file contains all best fit parameters from Anglada-Escude 2016 for
Proxima Centauri b (Proxima b) and the hint of Proxima c.

Note: All quoted errors are 1 sigma
"""

#########################################
#
# Stellar parameters for Proxima Centauri
#
#########################################

# Spectral Type
SPEC_TYPE = "M5.5V"

# Stellar Radius [in solar units]
RSTAR = 0.141

# Stellar Radius Error Range
RSTAR_ERR = [0.120,0.162]

# Stellar Mass [in solar units]
MSTAR = 0.120

# Stellar Mass Error Range
MSTAR_ERR = [0.105,0.135]

# Stellar Luminosity [in solar units]
LSTAR = 0.00155

# Stellar Luminosity Error Range
LSTAR_ERR = [0.00149,0.00161]

# Stellar Effective Temperature [K]
TEFF = 3050.

# Stellar Effective Temperature Error Range
TEFF_ERR = [2950.,3150.]

# Stellar Rotation Period [days]
ROTRAT = 83.

# System Age [yr] (Barnes et al 2016)
AGE = 4.8e9

# System Age Error Range
AGE_ERR = [3.7e9,7e9] # 6.2 Gyr from Barnes et al 2016, but we'll bump it up to 7 

#########################################
#
# Parameters for Proxima Centauri b
#
#########################################

# Planetary Period [days]
B_PERIOD = 11.186

# Planetary Period Error Range
B_PERIOD_ERR = [11.184,11.187]

# Doppler Amplitude [m/s]
B_DOP_AMP = 1.38

# Doppler Amplitude Error Range
B_DOP_AMP_ERR = [1.17,1.59]

# Orbital Eccentricity Range
B_ECC_RANGE = [0,0.35]

# Mean Longitude [degrees]
B_MEAN_LONG = 110.

# Mean Longitude Error Range
B_MEAN_LONG_ERR = [102.,118.]

# Argument of Periastron [degrees]
B_ARG_PERI = 310.

# Argument of Periastron Error Range
B_ARG_PERI_ERR = [0.,360.]

### Derived quantities below... ###

# Derived Semi-major Axis [AU]
B_SEMI = 0.0485

# Derived Semi-major Axis Error Range
B_SEMI_ERR = [0.0434,0.0526]

# Derived Minimum Mass, msini [in Earth masses]
B_MSINI = 1.27

# Derived Minimum Mass Error Range
B_MSINI_ERR = [1.10,1.46]