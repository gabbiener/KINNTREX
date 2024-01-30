#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 15:38:23 2022

@author: biener
"""

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import copy
import mAECell_Utils as utils

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Plots the difference electron density map (ground Truth) along with the 
# difference electron density map predicted by the AEPIRNN as a function of voxel index.
def plot_calcElecDens(Ground_Truth, Predicted, iFrame, Min, Max, yLimits, Save_Plot, GT_Scatter):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(range(Min,Max),Ground_Truth[iFrame, Min:Max], s=5, c='b')
    ax.plot(range(Min,Max),Predicted[Min:Max], s=5, c='r')
    plt.ylim(yLimits)
    fig.suptitle('Electron Density Difference for Time Points ' + str(iFrame+1), fontsize=18)
    plt.xlabel('Position index', fontsize=14)
    plt.ylabel('El. Dens. Diff.', fontsize=14)
    plt.show()
    if Save_Plot:
        fig.savefig('./data/Images/EDD_TimePoint_' + str(iFrame+1) + '.jpg')
        
def plot_Rates (File_Name, nEpoch, kLim, Save_Plot):
    K  = np.loadtxt(File_Name, delimiter=',')
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    fig.suptitle('Rates vs Epoch', fontsize=18)
    Legend_String = []
    for i in range(K.shape[0]):
        plt.plot(range(1,nEpoch+1),np.exp(K[i,:nEpoch]))
        Legend_String = Legend_String + ['k' + str(i+1)] 
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Rates', fontsize=14)
    plt.ylim(kLim[0], kLim[1])
    plt.legend(Legend_String)
    plt.show()
    if Save_Plot:
        fig.savefig('./data/Images/PINN_Weihts.jpg')
    return np.exp(K)

def plot_Conc (Conc, nEpoch, Plot_Epoch, Ground_Truth, TP, Save_Plot):
    fig, ax  = plt.subplots(1, 1, figsize=(7, 5))
    fig.suptitle('Weights vs Epoch', fontsize=18)
    fitRange = np.arange(TP[0], TP[-1]+0.003/2, 0.003)
    im       = plt.plot(fitRange,Conc[0,:])
    plt.scatter(TP[1:],Ground_Truth[0,:])
    for i in range(1,Conc.shape[0]):
        plt.plot(fitRange,Conc[i,:])
    for i in range(1,Ground_Truth.shape[0]):
        plt.scatter(TP[1:],Ground_Truth[i,:])
    plt.xlabel('log(Time)', fontsize=14)
    plt.ylabel('C', fontsize=14)
    plt.show()
    if Save_Plot:
        fig.savefig('./data/Images/PINN_Weihts.jpg')
    return im,

def plot_Loss (File_Name, nEpoch, yLim, Save_Plot):
    Loss  = np.loadtxt(File_Name, delimiter=',')
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    fig.suptitle('Loss vs Epoch', fontsize=18)
    plt.plot(range(1,nEpoch+1),Loss[:nEpoch])
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.ylim(yLim[0], yLim[1])
    plt.show()
    if Save_Plot:
        fig.savefig('./data/Images/Loss.jpg')
    return Loss

def Make_aMovie(K, TPoints, cGT, Model_Mat):
    # creating a blank window
    # for the animation
    
    c0     = torch.tensor([0.2, 0.0, 0.0, 0.0])
    k      = np.log(K)
    k      = torch.from_numpy(k).float()
    tT     = np.arange(TPoints[0], TPoints[-1]+0.003/2, 0.003)
    
    fig    = plt.figure()
    frame  = 0
    Conc_t = utils.Analytic_ODE_Solv (k[:,frame], np.exp(tT), c0, Model_Mat)
    im     = plot_Conc(Conc_t, 3000, frame, cGT, TPoints, False)
 
    # animation function
    def animate(frame, im, k, cGT, tT, TPoints, c0, Model_Mat):
        # appending values to the previously
        # empty x and y data holders
        Conc_t = utils.Analytic_ODE_Solv (k[:,frame], np.exp(tT), c0, Model_Mat)
        im     = plot_Conc(Conc_t, 3000, frame, cGT, TPoints, False)
        frame += 1
     
        return im,
 
    # calling the animation function    
    anim = animation.FuncAnimation(fig, animate, k.shape[1],
                                   fargs=(im, k, cGT, tT, TPoints, c0, Model_Mat),
                                   blit = True)
 
    # saves the animation in our desktop
    anim.save('growingCoil.mp4', writer = 'ffmpeg', fps = 2)