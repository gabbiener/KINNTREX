#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 17:15:11 2022

@author: biener
"""

import torch
import sys
import numpy as np
from NN_Rslt_Anal import NN_Rslt_Anal as Ana

Ana_O = Ana()

for ii in range(1,len(sys.argv)):
    curVar = sys.argv[ii]
    if '.inp' in curVar:
        TS_fName = curVar
    elif 'K_' in curVar:
        File_Name = curVar
    elif 'C_' in curVar:
        OutputFile = curVar
    else:
        CProf_fName = curVar

loaded_array      = np.loadtxt(TS_fName, usecols=0, dtype='str')
nTPoints          = int(loaded_array[0])
Ana_O.tPoints     = loaded_array[1:nTPoints + 1].astype(float)
Ana_O.loaded_Conc = np.loadtxt(CProf_fName)
Ana_O.K           = np.exp(np.loadtxt(File_Name, delimiter = ','))

Scale = 0.1537831
Conc_Interp, _, Conc_GT, _ = Ana_O.genConc(selIter = -1, c0 = torch.tensor([Scale, 0.0, 0.0, 0.0]), 
                                     dTau = 0.003, c2logk = False)