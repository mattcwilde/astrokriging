# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 14:53:26 2016

@author: dflemin3

This script extracts data from a VPLANET simulation suite directory.
"""

from __future__ import print_function, division, unicode_literals
import os

# Imports
from bigplanet import data_extraction as de

# Define root dirctory where all sim sub directories are located
src = "/astro/store/gradscratch/tmp/deitrr/davebot_newoct19/vplanet"

# Path to the hdf5 dataset
dataset= os.path.join(src,"proc_sims.hdf5")

# How you wish the data to be ordered (grid for grid simulation suites)
order = "none"

# Format of the data (default)
fmt = "hdf5"

# Ignore simulations that halted at some point?
remove_halts = False

# Any bodies whose output you wish to ignore?
skip_body = None

# Any parameters to extract from the log file?
var_from_log = {"star" : ["Mass"], 
                "b" : ["Mass","TidalQ"],
                "c" : ["Mass", "SemiMajorAxis"]}

# An optional kwarg that has extract_data_hdf5 output which simulation it's on
# every cadence steps for int cadence
cadence = 1000

# Compression algorithm to use
compression = "gzip"

data = de.extract_data_hdf5(src=src, dataset=dataset, order=order, cadence=cadence,
                            remove_halts=remove_halts, compression=compression,
                            var_from_log=var_from_log)
