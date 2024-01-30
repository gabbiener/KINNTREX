# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 16:28:30 2023

@author: biener
"""

import torch
import numpy as np
import mAECell_Utils as utils
from NN_Rslt_Anal import NN_Stat as stat
from NN_Rslt_Anal import NN_Rslt_Anal as Ana

relu1       = utils.ReLU1()
        
Ana_O       = Ana()
NN_Stat_O   = stat()

selIter     = -1
ScaleFName  = utils.fOpen(Title = 'Load Scale')
Scale_Array = np.loadtxt(ScaleFName)
Scale       = Scale_Array[selIter]
c0          = torch.tensor([Scale, 0.0, 0.0, 0.0])
dTau        = 0.003
Conc_Interp, tT, Conc_GT, tPoints = Ana_O.genConc(selIter = selIter, c0 = c0, dTau = dTau, c2logk = False)

Conc_Interp    = relu1(torch.from_numpy(Conc_Interp).double())
Sampled_Conc,_ = utils.getConc(tPoints, tT, Conc_Interp[1:,:])

NN_Stat_O.Sampled_Data = Sampled_Conc
WeiRes = NN_Stat_O.Weight_Res(Conc_GT[1:,:], Conc_GT[0,:], True, True)
print(WeiRes)

nInterm  = 3
strFName = utils.fOpen(Title = 'Load DED Maps with coordinates')
gtE      = np.loadtxt(strFName)
strFName = utils.fOpen(Title = 'Load Intermediate DED maps')
I        = np.loadtxt(strFName, delimiter=',')
I_Tensor = torch.from_numpy(np.reshape(I,(I.shape[0], -1, nInterm)))

calcE    = torch.matmul(I_Tensor[:, selIter, :].double(), Sampled_Conc[:-1,:].double())
NN_Stat_O.Sampled_Data = calcE

Res = NN_Stat_O.Residual(gtE[:,1:], gtE[:,0], False, False)
print(Res**2)

WeiRes = NN_Stat_O.Weight_Res(gtE[:,1:], gtE[:,0], False, False)
print(WeiRes)
# Res_FName = utils.fSaveAs(Title='Save Residual Files')
# np.savetxt(Res_FName, Res, delimiter=',', fmt='%1.7f')