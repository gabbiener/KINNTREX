# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 10:30:41 2023

@author: biener
"""

import numpy as np
from NN_Rslt_Anal import NN_Stat as stat

NN_Stat_O = stat()

Multi_fParmas_Array = NN_Stat_O.Assemble_Data()
Bins_Range          = np.arange(-10, 10, 0.1)
Bins_Range          = np.exp(Bins_Range)
counts, bins        = NN_Stat_O.genHistogram(Bins_Range = Bins_Range)
NN_Stat_O.Plot_Histogram(counts[4,:], bins, 'xlog')