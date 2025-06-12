# -*- coding: utf-8 -*-
"""
Created on Mon May 26 17:53:14 2025

@author: mateo
"""

import numpy as np
# If the code is in a file called plot_zplane.py
from plot_zplane import zplane 
b = np.array([1, 1, 1, 1])
a = np.array([0, 0, 0, 0])
zplane(b,a)