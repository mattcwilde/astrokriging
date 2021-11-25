# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 15:14:37 2016

@author: dflemin3

This script parses data from the Q100 HDF5 file and caches it into a user-friendly
dataframe.

Big ole' HDF5 file located in the following directory:

/astro/store/gradscratch/tmp/deitrr/davebot_Q100/vplanet

"""

from __future__ import print_function, division, unicode_literals
import os

# Imports
from bigplanet import data_extraction as de
import proxb_prob as pp

################################################################
#
# Load in HDF5 data
#
################################################################

# Define root dirctory where all sim sub directories are located
src = "/astro/store/gradscratch/tmp/deitrr/davebot_newoct19/vplanet"
data_loc = "../Data"
cache_name = "proc_sims_3sig.pkl"

# Path to the hdf5 dataset
dataset= os.path.join(src,"proc_sims.hdf5")

data = de.extract_data_hdf5(src=src,dataset=dataset)

################################################################
#
# Parse dataset, make dataframe of initial conditions
#
################################################################
# Define the bodies and body variables to extract using a dictionary
bodies = {'b' : ["Time","Semim","Ecce","Inc","ArgP","LongA","RotPer","Obli",
                 "SurfEnFluxEqtide","Mass","TidalQ"],
          'c' : ["Time","Ecce","Inc","ArgP","LongA","Mass","SemiMajorAxis"],
          'star' : ["Time","RotPer","Mass"]}

# Define the new value (dataframe column) to produce for a given body.  The new column
# and how to calculate it are given as a dictionary for each body.

new_cols = {}
kw = {}

# Extract and save into a cache (or read from it if it already exists)
# Note ind=0 makes it clear that we want initial conditions stored for all non-new_cols variables
df = de.aggregate_data(data, bodies=bodies,new_cols=new_cols,
                       cache=os.path.join(data_loc,cache_name),**kw)

################################################################
#
# Add columns for whether or not certain conditions are met
# i.e. is Prox b's eccentricity within 3 sigma of accept at 
# sometime between 3 and 7 Gyr
#
################################################################

new_cols = {"b" : {"Ecc_Prob" : pp.ecc_target, "Semi_Prob" : pp.semi_target,
                   "Prob" : pp.semi_ecc_target, "Prob_Frac" : pp.semi_ecc_frac}}

# Define any keyword arguments functions might need
kw = {"ssigma" : 3, "esigma" : 3, "ecc_name" : "Ecce", "semi_name" : "Semim"}

# Add the new columns!
df = de.add_column(data, df=df, new_cols=new_cols, 
                   cache=os.path.join(data_loc,cache_name), **kw)
                  
print("Done!")
                   