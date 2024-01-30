#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main program running KINNTREX

This program accepts three inputs
input:
    Param_fName: parameter file name
    Save_Path: The path in which the program saves the out put information
    Sample_Name: Uses the sample name for picking the parameters that are relevant to that
                 particular sample which may include reaction rate coefficient boundaries,
                 time points, and the file location of the difference electron density (DED) maps.
Outputs:
        Loss: Loss value at each selected iteration.
        I: Intermedate DED values at each selected iteration.
        rCoef: Reaction rate coefficeint at each selected iteration.
        InitScaling: Initial concentration condition Scaling. The initial conditions are used 
                     in the coupled diferential eaquations.
        Weights and biases of the last iteration, including the optimization parameters.
                     
Execution:
          The program starts by reading the parameters file, then it reads the ground 
          truth time-dependent DED maps. The following step is to generate the KINNTREX 
          neural networks and initiate them, followed by selecting the optimizer with the 
          relevant parameters such as learning rate. In case the running of KINNTREX is 
          continuing a former execution it loads the parameters of the former executions 
          at the last iteration. The last steps is to run KINNTREX n times and save the 
          output data.
        
Created on Thu Sep  8 15:38:23 2022

@author: biener
"""

import torch                  # A neural network toolbox
import sys                    # Used for reading the input variable.
import numpy as np            # Array handling toolbox used for reading the DED maps from a text file.
import mAECell_Utils as utils # NN separate utilities
from Param_fromInFile import Param_fromInFile as lparm # load parameters module
from KINNTREX_Class import KINNTREX                    # NN main module
from timeit import default_timer as timer
from NN_Rslt_Anal import NN_Rslt_Anal as Ana           # output management module
# %% Run Program
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    torch.seed() 
    
    Params_fName = None # Input parameters file name
    Save_Path    = None # Path in which the data will be saved
    Sample_Name  = 'SEQ 2 - Open 10' # simulation or measurements sample name
    
    # For batch running, i.e. not running through spyder this loop is needed.
    # it checks all the variables added to the running commend in a console.
    for ii in range(1,len(sys.argv)):
        curVar = sys.argv[ii]
        if '.txt' in curVar:
            Params_fName = curVar
        elif ' - ' in curVar:
            Sample_Name = curVar
        elif curVar[-1] == '\\':
             Save_Path = curVar
        else:
             Save_Path = curVar + '\\'

    Params        = lparm() # defining Parameters object

    # loading the parameters from a parameters file into the Params object and to a separate Param_List 
    # thelater is a list data type.
    Param_List    = Params.Load_inFile(Input_fName = Params_fName)
    Params.Load_Boundaries(Param_List, Sample_Name)
    learning_rate = Params.learning_rate # Learning Rate
    #%% Load Time dependent DED maps
    DED_File_Name = Params.Load_DatafName(Param_List, Sample_Name)
    loaded_array  = np.loadtxt(DED_File_Name)
    gtE           = loaded_array[:,1:]
    # %% Setting up Neural Networks in KINNTREX.
    model     = KINNTREX(Input=gtE, Params=Params)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # If selected loading weights and biases from former running in order to continue running it further.
    if model.initDistType == 'File':
        if model.initWeight_fName is None:
            model.initWeight_fName = utils.fOpen(Title = 'Load Checkpoint')
        checkpoint = torch.load(model.initWeight_fName)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # %% Execute KINNTREX
    start = timer()
    Last_OutData, Sampled_OutData = model.Training(gtE, Optimizer=optimizer)
    end   = timer()
    print(f'Time-lapse: {(end - start)//60} min')
    Save_Path_CP, _ = utils.Save_NN_Model(model, optimizer)
    if Save_Path is None:
        Save_Path = Save_Path_CP + '\\'
    # %% Save NN inputs and outputs as well as latent state info and NN parameters.
    npLoss   = Sampled_OutData['Loss']
    npLoss_e = Sampled_OutData['Loss_e']
    npLoss_i = Sampled_OutData['Loss_i']
    npLoss_k = Sampled_OutData['Loss_k']
    npLoss_c = Sampled_OutData['Loss_c']
    nprCoef  = Sampled_OutData['Rate']
    npI      = Sampled_OutData['I']
    npIniSc  = Sampled_OutData['InitScale']

    fName_Dict = {'Path'   : Save_Path,
                  'STD'    : model.initSTD,
                  'lr'     : learning_rate, 
                  'sHidden': None,
                  'nRun'   : 1,
                  'nEpochs': model.num_epochs,
                  'iRate'  : 1,
                  'Sign'   : 'm'}
    utils.Save_NN_Info(model, fName_Dict, K=nprCoef, I=npI, Loss=npLoss, Loss_e=npLoss_e, Loss_i=npLoss_i,
                       Loss_k=npLoss_k, Loss_c=npLoss_c, InitScale=npIniSc)
    fName_Dict['Data_Type'] = 'K_'
    Output_fName = utils.genData_fName(fName_Dict)
    Ana_Obj      = Ana()
    Ana_Obj.transKMatrix(Output_fName, model.c2logk)