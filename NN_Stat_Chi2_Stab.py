# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 16:28:30 2023

@author: biener
"""

import numpy as np
import mAECell_Utils as utils
from NN_Rslt_Anal import NN_Stat as stat

NN_Stat_O = stat()

Chi2 = NN_Stat_O.Multi_Chi2()
Chi2_FName = utils.fSaveAs(Title='Save chi2 Files')
np.savetxt(Chi2_FName, Chi2, delimiter=',', fmt='%1.7f')