#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 17:15:11 2022

@author: biener
"""

import torch
from NN_Rslt_Anal import NN_Rslt_Anal as Ana

Ana_O = Ana()
Scale = 0.1968121
Conc_Interp, _, Conc_GT, _ = Ana_O.genConc(selIter = -1, c0 = torch.tensor([Scale, 0.0, 0.0, 0.0]), 
                                     dTau = 0.003, c2logk = False)