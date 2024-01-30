#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 15:38:23 2022

@author: biener
"""

import os
from typing import Optional
import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch import Tensor
import numpy as np
import copy
import tkinter.filedialog

def Save_NN_Model(Model, Optimizer, initWeight_fName: str=None):
    if initWeight_fName is None:
        initWeight_fName = fSaveAs(Title = 'Save NN Model Params to:')
    torch.save({
            'model_state_dict': Model.state_dict(),
            'optimizer_state_dict': Optimizer.state_dict(),
            }, initWeight_fName)
    
    return os.path.dirname(initWeight_fName), os.path.basename(initWeight_fName)

"""
# A Rectified Linear Unit (ReLU) class that bounds the input values between 0 and 1. 
# Negative values will change to 0 and values above 1 will change to 1. 
# Any value between 0 and 1 will conserve its value.
"""
class ReLU1(nn.Hardtanh):
    r"""Applies the element-wise function:
    .. math::
        \text{ReLU1}(x) = \min(\max(0,x), 1)
    Args:
        inplace: can optionally do the operation in-place. Default: ``False``
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    .. image:: ../scripts/activation_images/ReLU1.png
    Examples::
        >>> m = nn.ReLU1()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, inplace: bool = False):
        super(ReLU1, self).__init__(0., 1, inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str
    
def genData_fName (Param_Dict):
    fName  = Param_Dict['Path']
    fName += Param_Dict['Data_Type']
    fName += 'STD-{0}_'.format(Param_Dict['STD'])
    fName += 'lr-{0}_'.format(Param_Dict['lr'])
    if Param_Dict['sHidden'] is not None:
        for ii in range(len(Param_Dict['sHidden'])-2):
            fName += '{0}-'.format(Param_Dict['sHidden'][ii])
        fName += '{0}_'.format(Param_Dict['sHidden'][ii+1])
    fName += 'Ep-{0}to{1}.dat'.format((Param_Dict['nRun']-1)*Param_Dict['nEpochs']+1,Param_Dict['nRun']*Param_Dict['nEpochs'])
    return fName


def Calc_rMatrix(k: Tensor, Model_Mat: Tensor):
    rMatrix = torch.matmul(Model_Mat, k)
    Size_2  = Model_Mat.shape[0]
    rMatrix = torch.reshape(rMatrix, (round(Size_2**0.5), round(Size_2/(Size_2**0.5))))
    rMatrix = rMatrix
    return rMatrix

def Diagonal_KMatrix(k: Tensor, Model_Mat: Tensor, Sort: bool=True):
    k                = k.double()
    Rate_Mat         = Calc_rMatrix(k, Model_Mat.double())
    Lambda, Eig_Vect = torch.linalg.eig(Rate_Mat)
    if Sort:
        LambdaR_Sorted, Index = Lambda.real.sort(descending=True, stable=True)
        LambdaI_Sorted   = torch.index_select(Lambda.imag, 0, Index)
        Eig_Vect         = torch.index_select(Eig_Vect, 1, Index)
        Lambda           = LambdaR_Sorted + 1.j*LambdaI_Sorted

    return Lambda, Eig_Vect

def Analytic_ODE_Solv (k: Tensor, TPoints: Tensor, C_Init_Cond: Tensor,
                       Model_Mat: Tensor, LambdaIn: Tensor=None):
#    relu = nn.ReLU()
    if not torch.is_tensor(TPoints):
        TPoints = torch.from_numpy(TPoints).float()
    else:
        TPoints = TPoints.float()
    TP_Ones  = torch.ones_like(TPoints)
    
    TPoints  = torch.complex(TPoints.double(), TP_Ones.double()*0)
    C0       = torch.complex(C_Init_Cond.double(), C_Init_Cond.double()*0)
    Unit     = torch.complex(TP_Ones.double(), TP_Ones.double()*0)
    
    Lambda, Eig_Vect = Diagonal_KMatrix(k, Model_Mat, LambdaIn is not None)
    invEig_Vect      = torch.linalg.inv(Eig_Vect)
    z0               = torch.unsqueeze(torch.mv(invEig_Vect, C0),1)
    
    matLambda        = torch.unsqueeze(Lambda,1)
    matTPoints       = torch.unsqueeze(TPoints-TPoints[0],0)
    expPortion       = torch.exp(torch.matmul(matLambda,matTPoints))
    initCond_Portion = torch.matmul(z0,torch.unsqueeze(Unit,0))
    z                = expPortion*initCond_Portion
    
    Conc             = torch.real(torch.matmul(Eig_Vect,z))
    
    return Conc, Lambda

def getConc(TimePoints, cTimeStemp, cValue):
    if not isinstance(TimePoints, np.ndarray):
        TimePoints  = TimePoints.detach().numpy()
    nTimePoints = len(TimePoints)
    Conc        = torch.zeros((cValue.shape[0], nTimePoints,))
    TS          = []
    for ii in range(nTimePoints):
        difference_array = np.absolute(cTimeStemp - TimePoints[ii])
        indConcTS = difference_array.argmin()
        TS.append(indConcTS)
        Conc[:, ii] = cValue[:,indConcTS]
    return Conc, np.array(TS)

# Saves the weights and biases information of the AEPIRNN throughout the epochs of the training.
def Save_NN_Info(net, fName_Dict: dict,
                 K: Optional[np.array]      = None,
                 I: Optional[np.array]      = None,
                 Loss: Optional[np.array]   = None,
                 Loss_e: Optional[np.array] = None,
                 Loss_i: Optional[np.array] = None,
                 Loss_B: Optional[np.array] = None,
                 Loss_k: Optional[np.array] = None,
                 Loss_c: Optional[np.array] = None,
                 InitScale: Optional[np.array] = None,
                 Format: Optional[str]      = '%1.7f'):
        
    if K:
        stackK  = np.stack(K,1)
        rstackK = stackK.reshape(stackK.shape[0], -1)
        fName_Dict['Data_Type'] = 'K_'
        fName        = genData_fName(fName_Dict)
        np.savetxt(fName, rstackK, delimiter=',', fmt=Format)
        
    if I:
        stackI  = np.stack(I,2)
        rstackI = np.moveaxis(stackI, [0,1,2], [0,2,1])
        rstackI = rstackI.reshape(rstackI.shape[0], -1)
        fName_Dict['Data_Type'] = 'I_'
        fName        = genData_fName(fName_Dict)
        np.savetxt(fName, rstackI, delimiter=',', fmt=Format)

    if Loss:
        stacknpLoss  = np.array(Loss)
        fName_Dict['Data_Type'] = 'Loss_'
        fName        = genData_fName(fName_Dict)
        np.savetxt(fName, stacknpLoss, delimiter=',', fmt=Format)
        
    if Loss_e:
        stacknpLoss  = np.array(Loss_e)
        fName_Dict['Data_Type'] = 'Loss_e_'
        fName        = genData_fName(fName_Dict)
        np.savetxt(fName, stacknpLoss, delimiter=',', fmt=Format)
        
    if Loss_i:
        stacknpLoss  = np.array(Loss_i)
        fName_Dict['Data_Type'] = 'Loss_i_'
        fName        = genData_fName(fName_Dict)
        np.savetxt(fName, stacknpLoss, delimiter=',', fmt=Format)
    
    if Loss_B:
        stacknpLoss  = np.array(Loss_B)
        fName_Dict['Data_Type'] = 'Loss_B_'
        fName        = genData_fName(fName_Dict)
        np.savetxt(fName, stacknpLoss, delimiter=',', fmt=Format)
    
    if Loss_k:
        stacknpLoss  = np.array(Loss_k)
        fName_Dict['Data_Type'] = 'Loss_k_'
        fName        = genData_fName(fName_Dict)
        np.savetxt(fName, stacknpLoss, delimiter=',', fmt=Format)
    
    if Loss_c:
        stacknpLoss  = np.array(Loss_c)
        fName_Dict['Data_Type'] = 'Loss_c_'
        fName        = genData_fName(fName_Dict)
        np.savetxt(fName, stacknpLoss, delimiter=',', fmt=Format)
        
    if InitScale:
        stacknpIniSc  = np.array(InitScale)
        fName_Dict['Data_Type'] = 'InitScale_'
        fName        = genData_fName(fName_Dict)
        np.savetxt(fName, stacknpIniSc, delimiter=',', fmt=Format)
        
def fOpen(Title: str = 'Open', File_Name: str=None):
    if File_Name is None:
        root      = tkinter.Tk()
        root.withdraw()
        File_Name = tkinter.filedialog.askopenfilename(title=Title)
    strFName  = str(File_Name)
    return strFName

def fSaveAs(Title: str = 'SaveAs', File_Name: str=None):
    if File_Name is None:
        root      = tkinter.Tk()
        root.withdraw()
        File_Name = tkinter.filedialog.asksaveasfilename(title = Title)
    strFName  = str(File_Name)
    return strFName

def Print_RRCs(k: Tensor=None):
    Sub1 = '\N{subscript one}: '
    Sub2 = '\N{subscript two}: '
    Sub3 = '\N{subscript three}: '
    Sub4 = '\N{subscript four}: '
    Sub5 = '\N{subscript five}: '
    SubM = 'k\N{subscript minus}'
    print (f'k' +\
           Sub1 + f' {k[0]:.2f}, k' + Sub2 + f' {k[1]:.2f}, k' + Sub3 + f' {k[2]:.2f}, k' +\
           Sub4 + f'{ k[3]:.2f}, k' + Sub5 + f' {k[4]:.2f}, \n' + SubM + Sub1 + f'{k[5]:.2f}, ' +\
           SubM + Sub2 + f'{k[6]:.2f}, ' + SubM + Sub3 + f'{k[7]:.2f}, ' +\
           SubM + Sub4 + f'{k[8]:.2f}, ' + SubM + Sub5 + f'{k[9]:.2f}')

def Print_RelaxRate_NN(RR: Tensor=None):
    Sub1 = '\N{subscript one}: '
    Sub2 = '\N{subscript two}: '
    Sub3 = '\N{subscript three}: '
    Sub4 = '\N{subscript four}: '
    print (f'\N{GREEK SMALL LETTER LAMDA}' +\
           Sub1 + f'{RR[0]:.4f}, \N{GREEK SMALL LETTER LAMDA}' +\
           Sub2 + f'{RR[1]:.4f}, \N{GREEK SMALL LETTER LAMDA}' +\
           Sub3 + f'{RR[2]:.4f}, \N{GREEK SMALL LETTER LAMDA}' +\
           Sub4 + f'{RR[3]:.4f}')