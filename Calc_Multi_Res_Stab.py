# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 16:28:30 2023

@author: biener
"""

import numpy as np
from NN_Rslt_Anal import NN_Stat as stat
import mAECell_Utils as utils

NN_Stat_O = stat()

WeiRes = NN_Stat_O.Multi_Res(Weighted_Res = True)
Res_FName = utils.fSaveAs(Title='Save weighted residual files')
np.savetxt(Res_FName, WeiRes, delimiter=',', fmt='%1.7f')

Res = NN_Stat_O.Multi_Res_DED(Weighted_Res = False)
Res_FName = utils.fSaveAs(Title='Save residual files')
np.savetxt(Res_FName, Res**2, delimiter=',', fmt='%1.7f')