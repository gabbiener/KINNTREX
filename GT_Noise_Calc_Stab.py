# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 23:22:31 2023

@author: biener
"""

import torch
import numpy as np
from NN_Rslt_Anal import NN_Stat as stat
import mAECell_Utils as utils

NN_Stat_O = stat()

GT_File      = utils.fOpen(Title='Load ground truth')
GT_plusCoord = np.loadtxt(GT_File)
GT           = GT_plusCoord[:,1:]
GT           = torch.from_numpy(GT).double()
GT_Noise = NN_Stat_O.Calc_Noise(GT)
print(GT_Noise.var())