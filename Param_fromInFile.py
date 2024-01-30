#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:55:35 2023

@author: biener
"""

import torch
from torch import Tensor
import numpy as np
import mAECell_Utils as utils
#%%
class Param_fromInFile:
    Model_Mat:        torch.Tensor = None
    nTPoints:         int = None
    nInterms:         int = None
    initSTD:          float = 0.02
    learning_rate:    float = 0.0001
    dTau:             float = None
    num_epochs:       int = 300000
    IProjCoeff:       float = 1.0e-13
    ConcCoeff:        float = 1.0
    LambdaLossCoeff:  float = 1.0e-4
    useExpBound:      bool = False
    incDark:          bool = False
    Second_LeakyReLU: bool = False
    logC_inConvNN:    bool = False
    c2logk:           bool = True
    LossToll:         float = 1.0e-6
    Repeat_LossToll:  int = 1000
    Percent_Repeat_LossToll: float = None
    B_Values:         torch.Tensor = None
    initCond:         torch.Tensor = torch.tensor([1.0, 0.0, 0.0, 0.0])
    nHidden_CB:       np.array = [189]
    nHidden_BK:       np.array = [64, 64, 64]
    nHidden_KB:       np.array = [64, 64, 64]
    nHidden_CK:       np.array = [189]
    NN_Block_List:    list = ['c2k']
    LB:               torch.Tensor = None
    HB:               torch.Tensor = None
    tPoints:          torch.Tensor = None

    def __init__(self, Model_Mat: torch.Tensor=None, nTPoints: int=None, nInterms: int=None, 
                 initSTD: float=None, learning_rate: float=None, dTau: float=None, num_epochs: int=None, 
                 IProjCoeff: float=1.0e-13, ConcCoeff: float=1.0, LambdaLossCoeff: float=1.0e-4, 
                 useExpBound: bool=False, incDark: bool=False, Second_LeakyReLU: bool=False, 
                 logC_inConvNN: bool=False, c2logk: bool=True, LossToll: float=1.0e-6, 
                 Repeat_LossToll: int=1000, Percent_Repeat_LossToll: float=None, B_Values: Tensor=None, 
                 initCond: Tensor=torch.tensor([1.0, 0.0, 0.0, 0.0]), nHidden_CB: np.array=[189], 
                 nHidden_BK: np.array=[64,64,64], nHidden_KB: np.array=[64,64,64], nHidden_CK: np.array=[189], 
                 NN_Block_List: list=['c2k']):
        self.Model_Mat        = Model_Mat
        self.nTPoints         = nTPoints
        self.nInterms         = nInterms
        self.initSTD          = initSTD
        self.learning_rate    = learning_rate
        self.dTau             = dTau
        self.num_epochs       = num_epochs
        self.IProjCoeff       = IProjCoeff
        self.ConcCoeff        = ConcCoeff
        self.LambdaLossCoeff  = LambdaLossCoeff
        self.useExpBound      = useExpBound
        self.incDark          = incDark
        self.Second_LeakyReLU = Second_LeakyReLU
        self.logC_inConvNN    = logC_inConvNN
        self.c2logk           = c2logk
        self.initCond         = initCond
        self.B_Values         = B_Values
        self.LossToll         = LossToll
        self.Repeat_LossToll  = Repeat_LossToll
        self.Percent_Repeat_LossToll = Percent_Repeat_LossToll
        self.nHidden_CB       = nHidden_CB
        self.nHidden_BK       = nHidden_BK
        self.nHidden_KB       = nHidden_KB
        self.nHidden_CK       = nHidden_CK
        self.NN_Block_List    = NN_Block_List
    
    def Load_Parameters(self, Input_fName: str = None):
        if Input_fName is None:
            Input_fName = utils.fOpen(Title = 'Load Parameters File')
        with open(Input_fName) as f:
            Parameters = f.readlines()
        return Parameters
        
    def Load_inFile(self, Input_fName: str = None):
        if Input_fName is None:
            Input_fName = utils.fOpen(Title = 'Load Parameters File')
        with open(Input_fName) as f:
            Parameters = f.readlines()
        sIndex = [Parameters.index(stSample) for stSample in Parameters if 
                  '[Params]' in stSample]
        sIndex = sIndex[0]
        eIndex = [Parameters.index(edSample) for edSample in Parameters if 
                  '[End Params]' in edSample]
        eIndex = eIndex[0]
        sIndex = sIndex + 1
        Params = Parameters[sIndex:eIndex]
        for ii in range(len(Params)):
            cParam = Params[ii].rstrip().split(',')
            if cParam[0]=='useExpBound':
                setattr(self, cParam[0], cParam[1].lstrip().rstrip().lower() == 'true')
            elif cParam[0]=='incDark':
                setattr(self, cParam[0], cParam[1].lstrip().rstrip().lower() == 'true')
            elif cParam[0]=='Second_LeakyReLU':
                setattr(self, cParam[0], cParam[1].lstrip().rstrip().lower() == 'true')
            elif cParam[0]=='logC_inConvNN':
                setattr(self, cParam[0], cParam[1].lstrip().rstrip().lower() == 'true')
            elif cParam[0]=='c2logk':
                setattr(self, cParam[0], cParam[1].lstrip().rstrip().lower() == 'true')
            elif cParam[0]=='initCond':
                setattr(self, cParam[0], torch.as_tensor([float(i) for i in cParam[1:]]))
            elif cParam[0]=='B_Values':
                setattr(self, cParam[0], torch.as_tensor([float(i) for i in cParam[1:]]))
            elif cParam[0]=='nHidden_CB':
                setattr(self, cParam[0], torch.as_tensor([int(i) for i in cParam[1:]]))
            elif cParam[0]=='nHidden_BK':
                setattr(self, cParam[0], torch.as_tensor([int(i) for i in cParam[1:]]))
            elif cParam[0]=='nHidden_KB':
                setattr(self, cParam[0], torch.as_tensor([int(i) for i in cParam[1:]]))
            elif cParam[0]=='nHidden_CK':
                setattr(self, cParam[0], torch.as_tensor([int(i) for i in cParam[1:]]))
            elif cParam[0]=='NN_Block_List':
                Values = [x.strip(' ') for x in cParam[1:]]
                setattr(self, cParam[0], [i for i in Values])
            elif 'E-' in cParam[1].lstrip().rstrip().lower():
                setattr(self, cParam[0], float(cParam[1]))
            elif 'e-' in cParam[1].lstrip().rstrip().lower():
                setattr(self, cParam[0], float(cParam[1]))
            elif '.' in cParam[1].lstrip().rstrip().lower():
                setattr(self, cParam[0], float(cParam[1]))
            elif any(map(str.isalpha, cParam[1].lstrip().rstrip())):
                setattr(self, cParam[0], cParam[1].lstrip().rstrip())
            else:
                setattr(self, cParam[0], int(cParam[1]))
        
        self.Model_Mat = self.Load_ModelMat(Parameters)
        
        return Parameters
    
    def Load_ModelMat(self, Param_List: list):
        siModelMat = [Param_List.index(Model_Mat_S) for Model_Mat_S in 
                      Param_List if "[Model Mat]" in Model_Mat_S]
        siModelMat = siModelMat[0]
        eiModelMat = [Param_List.index(Model_Mat_E) for Model_Mat_E in 
                      Param_List if "[End Model Mat]" in Model_Mat_E]
        eiModelMat = eiModelMat[0]
        siModelMat = siModelMat + 1

        if self.incDark:
            nRates = (self.nInterms-1)*(self.nInterms)-2
        else:
            nRates = (self.nInterms)*(self.nInterms+1)-2

        Model_Mat = torch.zeros([eiModelMat - siModelMat, nRates])
        for i, Line in enumerate(Param_List[siModelMat:eiModelMat]):
            LineFloat = np.fromstring(Line, dtype=float, sep=',')
            Model_Mat[i,:] = torch.from_numpy(LineFloat)
        return Model_Mat
    
    def Load_Boundaries(self, Param_List: list, Sample_STR: str):
        Index = [Param_List.index(stSample) for stSample in Param_List if 
                  Sample_STR in stSample]
        Sample_Param = Param_List[Index[0] + 1:Index[1]]

        Par    = []
        for ii in range(len(Sample_Param)):
            if not Sample_Param[ii].lstrip().rstrip().split(',',1) == ['']:
                Par.append(Sample_Param[ii].lstrip().rstrip().split(',',1))
        Par    = list(zip(*Par))
        zroLB  = float(Par[1][Par[0].index('LB_Zero')])
        zroHB  = float(Par[1][Par[0].index('HB_Zero')])
        ocLB   = np.array(Par[1][Par[0].index('LB_OC')].split(',')).astype(float)
        ocHB   = np.array(Par[1][Par[0].index('HB_OC')].split(',')).astype(float)
        LB     = np.array(Par[1][Par[0].index('LB')].split(',')).astype(float)
        HB     = np.array(Par[1][Par[0].index('HB')].split(',')).astype(float)
        self.LB = torch.from_numpy(LB + (1 - ocLB)*zroLB)
        self.HB = torch.from_numpy(HB + (1 - ocHB)*zroHB)
        self.tPoints = self.Load_tPoints(Param_List, Sample_STR)
        
        return self.LB,self.HB
    
    def Load_tPoints (self, Param_List: list, Sample_STR: str):
        Index = [Param_List.index(stSample) for stSample in Param_List if Sample_STR
                 in stSample]
        Sample_Param = Param_List[Index[0] + 1:Index[1]]

        ind   = [Sample_Param.index(ii) for ii in Sample_Param if 'TimeStamp' in ii]
        tPoints = np.array(Sample_Param[ind[0]][:-1].split(',')[1:]).astype(float)
        
        return torch.from_numpy(tPoints)
    
    def Load_DatafName (self, Param_List: list, Sample_STR: str):
        Index = [Param_List.index(stSample) for stSample in Param_List if Sample_STR
                 in stSample]
        Sample_Param = Param_List[Index[0] + 1:Index[1]]

        ind     = [Sample_Param.index(ii) for ii in Sample_Param if 'ElecDens' in ii]
        Data_FN = Sample_Param[ind[0]][:-1].split(',')[1].lstrip()
        
        return Data_FN