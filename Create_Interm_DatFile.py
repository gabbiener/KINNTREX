#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 23:37:56 2022

@author: biener
"""

import numpy as np
import mAECell_Utils as utils
#%%
selEpoch = -1
nInterm  = 3
strFName = utils.fOpen(Title = 'Load DED Maps with coordinates')
ED       = np.loadtxt(strFName)
strFName = utils.fOpen(Title = 'Load Intermediate DED maps')
I        = np.loadtxt(strFName, delimiter=',')
I        = np.reshape(I,(I.shape[0], -1, nInterm))
#%%
selI     = I[:,selEpoch,:]
for ii in range(nInterm):
    currI    = selI[:,ii]
    currI    = np.dstack((ED[:,0],currI)).squeeze()
    strFName = utils.fSaveAs(Title = 'Save intermedate dat file')
    np.savetxt(strFName[:-4] + str(ii+1) + '.dat', currI, delimiter=' ')