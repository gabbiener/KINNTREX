#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file define the KINNTREX class that include 
Parameters:
           cLSV_W: Weight matrix multiplying left singular vectors data to get the 
                   intermediates DED maps.
           Conc_W: Weights matrix multiplying the intermedates DED layer to recalculate the 
                   time-dependednt DED maps. This matrix is also considered as the 
                   concentration matrix
           Interm: Holds the intermediates DED values.
           LB: Low boundary for the rate coefficients ranges
           HB: high boundary for the rate coefficients ranges
           initScale: Initial scale of the initial conditions for the concentrations.
           Additional parameters are explain within the init method.
Methods:
        __init__: Initiation of KINNTREX
        init_Weights: initiating weights and bases within KINNTREX
        Save_NN_Model: Saving weights and biases of the last iteration.
        Load_NN_Model: Loading the weights and biases from the last iteration of a former run to initiate a consequtive run.
        genSub_NN: Generate a neural network based on number of neurons in the input, output, hidden layers.
        PINN_Loss: Calculate the loss value cmparing the calculated DED and the input DED, calculated concentrations 
                   before and after coupled differential equations, comparing the intermediate
                   of the dark state to zero, making sure the RRCs won't go over the boundaries
                   set by the user.
        Projection_NN: calculates the intermediates DED maps in an intermediate layer as well as the concentrations as weights and time-dependent DED mpas at the output layer.
        C2k_NN: This method adds a fully connected neural network to KINNTREX with the concentration as input and reaction rate coefficients as output.
        Diag_k2Lambda_NN: Diagonalizing reaction rate coefficent matix with relaxation rates as output
        k2Lambda_NN: method adds a fully connected neural network to KINNTREX with the reaction rate coefficients as input and relaxation rate as output.
        C2Lambda_NN: method adds a fully connected neural network to KINNTREX with the concentration as input and relaxation rate as output.
        Lambda2k_NN: method adds a fully connected neural network to KINNTREX with the relaxation rates as input and reaction rate coefficients as output.
        SolveCDE: solves the coupled differential equations governing the kinetics of the investigated molecule. The reaction rate coefficients are input and the resulting concentrations of the intermediates are the output.
        forward: Calculates the entire KINNTREX interation going foreward through the sub neural networks and solution of the CDE. It runs a single iteration.
        Training: This method runs the KINNTREX for the multiple interations and calls the forward method to calculate all the needed quantities. This method also runs the backpropagation optimization after calculating the loss value.
Created on Wed Oct 19 09:03:03 2022

@author: biener
"""
# from typing import Optional
import torch                    # imports all the neural network package of pytorch
import torch.nn as nn           # Names the nn class as nn for shortcut
from torch import Tensor        # Names the Tensor class for shortcut
from torch.nn.parameter import Parameter # Names the Parameter class for shortcut
import numpy as np              # Imports the numpy package that has matrix algebra and names it np for shortcut
import mAECell_Utils as utils   # imports the library including utilities for the KINNTREX class
from Param_fromInFile import Param_fromInFile as lparm # imports the library of parameters handling and names it lparm for shortcut

# The Kinetically Inspired Neural Network for Time-Resolved X-rays class
class KINNTREX(nn.Module):
    
    cLSV_W: Tensor    # Matrix A the weight used in the first sub-NN (Projection NN) for the calculation of the middle layer perceptrons
    Conc_W: Tensor    # Cnn Matrix, the weights used for the calculation of time dependent difference electron density maps (Ec1)
    Interm: Tensor    # Intermediate state perceptrons
    LB:     Tensor    # Lower Boundar of the RRCs
    HB:     Tensor    # Upper Boundar of the RRCs
    initScale: Tensor # Scaling the initial concentration between 0 and 1
    
    """
    Initialization method of the KINNTREX class
    This method initiates the weights and biases of all the sub networks drawing from uniformly distributed function of a gaussian distribution.
    This method also creates the sub-networks and updates the KINNTREX object parameters.
    input: Input - time dependent difference electron density maps, flattend and placed in a 2-D matrix
           Model_Mat - a matrix used to convert the reaction rate coefficients vector to a reactive rate coefficients matrix used in the differential equations.
           nInterms - number of intermediate states
           Params - Neural Network parameters loadded from a file at the initiation of the neural network.
    returns: self - instantiation of the KINNTREX calss
    """
    def __init__(self, Input: Tensor, Model_Mat: Tensor=None, nInterms: int=None, tPoints: Tensor=None, Params: lparm=None,
                 Sample_STR: str='SEQ 1 - Open 10'):
        super(KINNTREX, self).__init__()
        self.relu      = nn.ReLU()         # ReLU activation function with no upper limiit used in the class
        self.relu1     = utils.ReLU1()     # ReLU1 activatin function with the upper limit of the value 1 used in the class
        self.leakyrelu = nn.LeakyReLU(0.2) # leakyReLU activatin function with no upper limit used in the class
        
        if Params is None:
            Params     = lparm() # defining Parameters object
            
            # loading the parameters from a parameters file into the Params object and to a separate Param_List 
            # thelater is a list data type.
            Param_List = Params.Load_inFile(Input_fName = None)
            Params.Load_Boundaries(Param_List, Sample_STR) # lower and upper bound of reaction rate coefficents variability ranges.
            
        if tPoints is None:
            tPoints = Params.tPoints
            
        if nInterms is None:
            self.nInterms      = Params.nInterms          # Number of Intermediates
        else:
            self.nInterms      = nInterms
            
        self.initSTD          = Params.initSTD           # The STD used for the initiation of the weights for the encoder and the decoder sections.
        self.initDistType     = Params.initDistType      # Weights and biasses initiation type. Can be either 'Uniform' with the initSTD value as initiation value, 'Normal' for normal distribution using initSTD as standard deviation of the distribution, or loadded from a file, i.e. 'File'
        self.num_epochs       = Params.num_epochs        # number of iterations
        self.dTau             = Params.dTau              # increment value in log(time) scale
        self.IProjCoeff       = Params.IProjCoeff        # coefficient, determining importance of projection of intermediates onto each other in the loss function
        self.ConcCoeff        = Params.ConcCoeff         # Coefficient value determiniing importance of Concentration comparison between Cnn and Ccde in the loss function
        self.LambdaLossCoeff  = Params.LambdaLossCoeff   # Coefficient value determiniing the importance of relaxation rate comparison to the ground truth in the loss function
        self.useExpBound      = Params.useExpBound       # Switch between using exponential values for the RRC range bounderies or not.
        self.incDark          = Params.incDark           # include the dark state amnong the intermediates
        self.Second_LeakyReLU = Params.Second_LeakyReLU  # Switch between using the second leakyrelu for the RRCs or not.
        self.logC_inConvNN    = Params.logC_inConvNN     # Switch between using log(Cnn) or just Cnn
        self.c2logk           = Params.c2logk            # Switch between considering RRC from conversion NN to be in log scale or linear scale.
        self.initCond         = Params.initCond          # Initial conditions for the intermediate concentrations.
        self.B_Values         = Params.B_Values          # Relaxation rate values
        self.NN_Block_List    = Params.NN_Block_List     # Neural network sequantial blovk assembly
        self.LossToll         = Params.LossToll          # Loss value tollerance. stops the NN running after loss value reaches tollerance value.
        self.Repeat_LossToll  = Params.Repeat_LossToll   # NN algrithm stops after loss value reached tollerance values n times. n is Repeat_LossToll
        self.Percent_Repeat_LossToll = Params.Percent_Repeat_LossToll # NN algrithm stops after loss value reached tollerance values n% times. n is Percent_Repeat_LossToll. The precentage is from total number of iterations.
        self.LB               = Params.LB                # RRC ranges lower bound values
        self.HB               = Params.HB                # RRC ranges higher bound values
        self.initWeight_fName = None                     # File name of saved Neural network model.
        self.tPoints          = tPoints 
        
        nTime_Points       = Input.shape[1]              # number of time points for the time-dependent difference electron density maps
        
        self.initScale = Parameter(torch.ones(size=(1, ))*self.initCond[0]) # initiaiting the initial concentration scale.

        # initiating the weights for projection NN            
        if self.initDistType == 'Uniform':
            self.cLSV_W    = Parameter((torch.rand(size=(self.nInterms, self.nInterms))*2-1)*self.initSTD) # Registering the Weights parameters into the graph for the left singular vector projection series coefficients (Weights for layer 1).
            self.Conc_W    = Parameter((torch.rand(size=(self.nInterms, nTime_Points))*2-1)*self.initSTD) # Registering the Weights parameters into the graph for the Concentration Weight Values (Weights for layer 2).
        elif self.initDistType == 'Normal':
            self.cLSV_W    = Parameter(torch.normal(0.0, self.initSTD, size=(self.nInterms, self.nInterms)))
            self.Conc_W    = Parameter(torch.normal(0.0, self.initSTD, size=(self.nInterms, nTime_Points)))
        else:
            self.cLSV_W    = Parameter(torch.normal(0.0, self.initSTD, size=(self.nInterms, self.nInterms))) # Registering the Weights parameters into the graph for the left singular vector projection series coefficients (Weights for layer 1).
            self.Conc_W    = Parameter(torch.normal(0.0, self.initSTD, size=(self.nInterms, nTime_Points))) # Registering the Weights parameters into the graph for the Concentration Weight Values (Weights for layer 2).
            
        # calculating the number of reaction rate coefficients from the number of intermediates including the dark state
        if self.incDark:
            nCoeff = (self.nInterms-1)*(self.nInterms)-2
        else:
            nCoeff = (self.nInterms)*(self.nInterms+1)-2

        # number of items in the flattened concentration vector
        nConc_Flat     = nTime_Points*self.nInterms

        # Initiating the neural network blocks. Block 1, c2k, is a nn converting concentration to RRC.
        # Block 2, k2lambda is an NN converting RRCs to relaxation rates.
        # Block 3, c2lambda is an NN converting Concentrations to relaxation rates.
        # Block 4, lambda2k is an NN converting relaxation rates to RRCs.
        if 'c2k' in self.NN_Block_List:
            self.genSub_NN('nLayers_CK', 'Hidden_CK', nConc_Flat, Params.nHidden_CK, nCoeff)
        if 'k2lambda' in self.NN_Block_List:
            self.genSub_NN('nLayers_KB', 'Hidden_KB', nCoeff, Params.nHidden_KB, len(self.B_Values))
        if 'c2lambda' in self.NN_Block_List:
            self.genSub_NN('nLayers_CB', 'Hidden_CB', nConc_Flat, Params.nHidden_CB, len(self.B_Values))
        if 'lambda2k' in self.NN_Block_List:
            self.genSub_NN('nLayers_BK', 'Hidden_BK', len(self.B_Values), Params.nHidden_KB, nCoeff)
    
        # if Relaxation rates were loaded then modify RRC range boundaries such that lower limit is 0 and
        # upper limit is 1.2 times the highest relaxation rate values.
        if self.B_Values is None:
            if self.LB is None:
                self.LB = torch.tensor([1.0E-10])*torch.ones(nCoeff)
            if self.HB is None:
                self.HB = torch.tensor([1.0E+10])*torch.ones(nCoeff)
        else:
            self.LB = torch.maximum(self.LB, torch.tensor([1.0E-10])*torch.ones(nCoeff))
            self.HB = torch.minimum(self.HB, torch.max(self.B_Values)*1.2*torch.ones(nCoeff))

        # initiating weights and biases
        self.apply(self.init_weights)
        
        # preserving the values of Input and Model_Mat in the class (Object)
        self.Input     = Input
        if Model_Mat is None:
            self.Model_Mat = Params.Model_Mat
        else:
            self.Model_Mat = Model_Mat

    """    
    A method to initiate weights and biases of the neural network. The weights and biases 
    can be initiated by drawing values from either a uniform distribution or a Gaussian distribution
    input: apart from 'self' which is the KINNTREX class instansiation, the other input is the model
           which inour case is part of self. the part that is responsible for the neuron layers.
    returns: self - instantiation of the KINNTREX calss
    """
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            if self.initDistType == 'Uniform': # drawing from unifrom distribution with range [-Value, Value]. Value = initSTD
                module.weight.data.uniform_(-self.initSTD, self.initSTD)
            elif self.initDistType == 'Normal': # drawing from Gaussian distribution with mean 0 and STD=initSTD
                module.weight.data.normal_(mean=0.0, std=self.initSTD)
            else: # drawing from Gaussian distribution with mean 0 and STD=initSTD
                module.weight.data.normal_(mean=0.0, std=self.initSTD)
            if module.bias is not None:
                if self.initDistType == 'Uniform': # drawing from unifrom distribution with range [-Value, Value]. Value = initSTD
                    module.bias.data.uniform_(-self.initSTD, self.initSTD)
                elif self.initDistType == 'Normal': # drawing from Gaussian distribution with mean 0 and STD=initSTD
                    module.bias.data.normal_(mean=0.0, std=self.initSTD)
                else: # drawing from Gaussian distribution with mean 0 and STD=initSTD
                    module.bias.data.normal_(mean=0.0, std=self.initSTD)
        
    """
    A method to save the NN wiehgts and biases after multiple iterations
    input: self - instantiation of the KINNTREX calss
           Model_fName - File name to which the model will be save to
    returns: self - instantiation of the KINNTREX calss
    """
    def Save_NN_Model(self, Model_fName: str=None):
        if Model_fName is None:
            Model_fName = utils.fSaveAs(Title = 'Save Model into:')
        torch.save(self.state_dict(), Model_fName)
        
    """
    A method to load Weights and biases values from a .pth file
    inpput: self - instantiation of the KINNTREX calss
           Model_fName - File name from which the weights and biases are loadded
    returns: self - instantiation of the KINNTREX calss
    """
    def Load_NN_Model (self, Model_fName: str=None):
        if Model_fName is None:
            Model_fName = utils.fOpen(Title = 'Load Model File')
        self.load_state_dict(torch.load(Model_fName))
                    
    """
    a method for generic creation of linear layers of neurons for a sub neural network.
    input: self - instantiation of the KINNTREX calss
           nLayersSTR - the parameter name for the parameter holding the number of layers
           HiddenSTR - the parameter name for the parameter holding the actual NN construction
           ninput - number of neurons in the input layer
           nhidden - an array holding the number of neurons in each hidden layer.
           noutput - The number of neurons in the output layer
    returns: self - instantiation of the KINNTREX calss
    """
    def genSub_NN(self, nLayersSTR: str=None, HiddenSTR: str=None, ninput: int=None, nhidden: np.array=None, 
                  noutput: int=None):
        nHidden = np.hstack((ninput, nhidden))
        nHidden = np.hstack((nHidden, noutput))
        Hidden  = nn.ModuleList()
        for k in range(len(nHidden)-2):
            Hidden.append(nn.Linear(nHidden[k], nHidden[k+1]))
        Hidden.append(nn.Linear(nHidden[-2], nHidden[-1], bias=False))
        setattr(self, HiddenSTR, Hidden)
        setattr(self, nLayersSTR, len(nHidden))

    """
    The loss function method: calculates the loss value for KINNTREX. The loss value consists 
    of comparison between input time-dependent DED mpas and calculated DED maps, comparing
    between two sets of calculated concentration profiles, comparing dark state DED maps to 0,
    na dmonitoring reaction rate coefficeints, making sure they will not excceed their selected
    ranges.
    input: Calc1 - calculated time dependent DED maps
           Calc2 - recalculated time dependent DED maps
           GT - ground truth of the time dependent DED maps
           CalcC - Intermediates Concentration profile calculated by solving the differential equation govering protein photo cycle 
           k - Reaction rate coefficients vector
           LB - lower bound limit of the reaction rate coefficients ranges
           HB - higher bound limit of the reaction rate coefficients ranges
           IProjectCoeff - coefficient multiplying the projection loss. 0 means projection loss is not taken into account. The projection loss is calculated by projecting one intermidiate onto the others.
           ConcCoeff - coefficient multiplying the concentration loss. 0 means concentration loss is not taken into account. concentration loss is calculated by substructing Cnn (NN calculated concentration) from Code (concentration calculated by solving the differential equation)
           useExpBound - a boolian variable for using exponential boundaries for the reaction rate coefficients or regular boundaries.
    returns: loss - total loss value
             loss_e - loss related to comparing the input and calculated time-dependent DED 
                      maps
             loss_i - loss related to intermediate DED maps.
             loss_Lambda - loss related to relaxation rates
             loss_k - loss related to the reaction rate coefficients remaining with the ranges
                      boundaries
             loss_c - loss related to comparison between two calculated concentrations
             loss_is - loss related to initial scalingvalues for the intial condition.
    """
    def PINN_Loss(self, Calc1: Tensor, Calc2: Tensor, GT: Tensor, calcC: Tensor, k: Tensor,
                  Lambda_fromC: Tensor, Lambda_fromK: Tensor):
        
        # calculating L_E and L_C, i.e. time dependent DED map related loss as well as intermediate cocentration loss
        # calculates the difference between first time dependent DED maps and ground truths
        loss_e  = ((Calc1 - GT)**2).mean()
        # calculates the difference between second time dependent DED maps and ground truths
        loss_e += ((Calc2 - GT)**2).mean()
        
        # calculates the difference between intermediate concentration profile calculated using projection NN and cocnentrations calculated by solving the differential equations
        if self.incDark:
            loss_c  = ((self.Conc_W - calcC)**2).mean()
        else:
            loss_c = ((self.Conc_W - calcC[:-1,:])**2).mean()
        loss_c = loss_c*self.ConcCoeff

        # calculating the loss resulting from last intermediate(dark state) not beeing 0. dark state is used as the reference and thus the dark state DED is substructed from all the intermediate  
        if self.incDark:
            loss_i  = ((self.Interm[:,-1])**2).mean()
        else:
            loss_i = torch.tensor([0.0])
            
        # calculating the loss resulting from intermediate projecting onto each other (Intermediate disimilarity check). Shai's suggestion
        if self.incDark:
            nInterm = self.Interm.shape[1]-1
        else:
            nInterm = self.Interm.shape[1]
            
        for ii in range(nInterm):
            for jj in range(ii):
                normIi  = (self.Interm[:,ii]**2).sum()**0.5
                normIj  = (self.Interm[:,jj]**2).sum()**0.5
                loss_i += (torch.dot(self.Interm[:,ii]/normIi, self.Interm[:,jj]/normIj))**2*self.IProjCoeff
        
        # calculate the loss related to the relaxation rates while comparing to the ground truth
        if self.B_Values is not None:
            if Lambda_fromC is not None:
                loss_Lambda = ((torch.log(self.B_Values) - Lambda_fromC)**2).max()
            else:
                loss_Lambda = torch.tensor([0])
                
            loss_Lambda = loss_Lambda.to(torch.float32)
            if Lambda_fromK is not None:
                if self.c2logk and 'k2lambda' in self.NN_Block_List:
                    loss_Lambda += ((torch.log(self.B_Values[1:]) + Lambda_fromK[1:])**2).max()
                else:
                    nFactor_2 = torch.maximum(self.B_Values.max(), -Lambda_fromK.real.to(torch.float32).max())**2
                    loss_Lambda += ((self.B_Values + Lambda_fromK.real.to(torch.float32))**2).max()/nFactor_2
        else:
            loss_Lambda = torch.tensor([0])
            
        loss_Lambda = loss_Lambda*self.LambdaLossCoeff

        # calculate the loss related to the reaction rate coefficient having values within or out side the preset ranges.            
        if self.useExpBound:
            if self.c2logk:
                k = torch.exp(k)
            loss_k  = ((torch.minimum(k, self.LB) - self.LB)**2).mean()
            loss_k += ((torch.maximum(k, self.HB) - self.HB)**2).mean()
        else:
            if not self.c2logk:
                k = torch.log(self.relu(k)+1e-10)
                
            loss_k  = ((torch.maximum(k, torch.log(self.HB)) - torch.log(self.HB))**2).sum()
            loss_k += ((torch.minimum(k, torch.log(self.LB)) - torch.log(self.LB))**2).sum()
            
        loss_k = loss_k*1.0
        
        # Limiting the range of initial scale to be between 0 and 1
        loss_is  = (torch.minimum(self.initScale, torch.tensor([0])))**2
        loss_is += (torch.maximum(self.initScale, torch.tensor([1])) - torch.tensor([1]))**2
        
        # total loss value calculated.
        return loss_e + loss_i + loss_Lambda + loss_k + loss_c + loss_is, loss_e, loss_i, loss_Lambda,\
               loss_k, loss_c, loss_is

    """    
    A method generating the projection NN connecting significant lSVs to intermediates and intermediates to
    recalculated time dependent DED maps. Concentrations are also calculated as weights between
    intermediates layer and time dependent DED  maps
    input: self - instantiation of the KINNTREX calss
           LSV - significant left sinular vectors
    returns: I - Flattend Intermediate DED mpas, 
             CalcE1 - Flattend recalculated time dependent DED maps
    """
    def Projection_NN(self, LSV: Tensor):
        I      = torch.matmul(LSV.float(), self.cLSV_W)
        calcE1 = torch.matmul(I, self.Conc_W)
        
        return I, calcE1
    
    """
    A method generating the conversion NN connecting itermediate concentrations to
    reaction rate coefficients (RRCs).
    The method converts between concentrations and RRCs using a fully connected artificial neural network
    with a single or multiple hiddent layers (user can choose the number of layers and
    the number of neurons in each layer). 
    input: self - instantiation of the KINNTREX calss
    returns: k - RRCs.
    """
    def C2k_NN(self):

        # separating the concentration of intermediates calculated in the projection NN from its role as weight for the projection NN.
        # Separation is executed inorder to prevent interference with the NN operation treating the
        # cocnetration as weights.
        Conc      = torch.zeros_like(self.Conc_W)
        Conc.data = self.Conc_W.data
        Conc      = self.relu1(Conc)   # making sure the concentrations are limited between 0 and 1
        
        # If statement is executed in case user choose to take the log of the concentration 
        if self.logC_inConvNN:
            Conc = torch.log(Conc + 1e-10)
        
        # Actual Flattening is happening here
        c2k = torch.reshape(Conc, (-1,))
        for iLayer in range(self.nLayers_CK-2):
            c2k = self.Hidden_CK[iLayer](c2k)    # calculating next layer using former layer, weights, and biases
            c2k = self.leakyrelu(c2k)            # using leakyrelu activation function for nonlinear action
        k = self.Hidden_CK[-1](c2k)              # calculating output from last layer, weight and biases
        
        #If user chooses, second activation of leakyrelu function is performed on the output layer.
        if self.c2logk:
           if self.Second_LeakyReLU:
               k = self.leakyrelu(k)
            
        return k
    
    """
    A method calculating relaxation rates when RRCs ar given.
    The method uses reaction rate coefficient matrix diagonalization.
    input: self - instantiation of the KINNTREX calss
           k - RRCs.
    returns: Lambda - Relaxation rates.
    """
    def Diag_k2Lambda_NN(self, k: Tensor):
        Sort = True
        if self.c2logk:
            Lambda, _ = utils.Diagonal_KMatrix(torch.exp(k), self.Model_Mat, Sort)
        else:
            Lambda, _ = utils.Diagonal_KMatrix(k, self.Model_Mat, Sort)
        Lambda = Lambda.real

        return Lambda
    
    """
    A method calculating relaxation rates when RRCs ar given.
    The method converts between RRCs and realxation rates using a fully connected artificial neural network
    with a single or multiple hiddent layers (user can choose the number of layers and
    the number of neurons in each layer).
    input: self - instantiation of the KINNTREX calss
           k - RRCs.
    returns: Lambda - Relaxation rates.
    """
    def k2Lambda_NN(self, k: Tensor):
#        if self.c2logk:
#            k2Lambda = torch.exp(k)
#        else:
        k2Lambda = k
        for iLayer in range(self.nLayers_KB-2):
            k2Lambda = self.Hidden_KB[iLayer](k2Lambda) # calculating next layer using former layer, weights, and biases
            k2Lambda = self.leakyrelu(k2Lambda)
        Lambda = self.Hidden_KB[-1](k2Lambda)
        
        Lambda_Sorted, _ = Lambda.sort(descending=True, stable=True)

        return Lambda_Sorted
    
    """
    A method calculating relaxation rates when intermediate concentrations ar given.
    The method converts between concentrations and realxation rates using a fully connected artificial neural network
    with a single or multiple hiddent layers (user can choose the number of layers and
    the number of neurons in each layer).
    input: self - instantiation of the KINNTREX calss
    returns: Lambda - Relaxation rates.
    """
    def C2Lambda_NN(self):
        
        # separating the concentration of intermediates calculated in the projection NN from its role as weight for the projection NN.
        # Separation is executed inorder to prevent interference with the NN operation treating the
        # cocnetration as weights.
        Conc      = torch.zeros_like(self.Conc_W)
        Conc.data = self.Conc_W.data
        Conc      = self.relu1(Conc)
        
        # If statement is executed in case user choose to take the log of the concentration 
        if self.logC_inConvNN:
            Conc = torch.log(Conc + 1e-10)
        
        # Actual Flattening is happening here
        c2Lambda = torch.reshape(Conc.T, (-1,))
        for iLayer in range(self.nLayers_CB-2):
            c2Lambda = self.Hidden_CB[iLayer](c2Lambda) # calculating next layer using former layer, weights, and biases
            c2Lambda = self.leakyrelu(c2Lambda)
        Lambda = self.Hidden_CB[-1](c2Lambda)
        if self.Second_LeakyReLU:
            Lambda = self.leakyrelu(Lambda)

        return Lambda
    
    """
    A method calculating reaction rate coefficients when relaxation rates ar given.
    The method converts between realxation rates and rate coefficients using a fully connected
    artificial neural network with a single or multiple hiddent layers (user can choose the
    number of layers and the number of neurons in each layer).
    input: self - instantiation of the KINNTREX calss
    returns: k - reaction rate coefficients.
    """
    def Lambda2k_NN(self, Lambda: Tensor):
#        if self.c2logk:
#            Lambda2k = -torch.exp(-Lambda)
#        else:
        Lambda2k = Lambda
        for iLayer in range(self.nLayers_BK-2):
            Lambda2k = self.Hidden_BK[iLayer](Lambda2k) # calculating next layer using former layer, weights, and biases
            Lambda2k = self.leakyrelu(Lambda2k)
        k = self.Hidden_BK[-1](Lambda2k)
#        k = torch.log(self.relu(k)+1e-10)
        
        return k

    """
    A method solving the coupled differential equations governing the kinetics of thee 
    investigated molecule. This method also recalculates time-dependent DED maps using the
    extracted concentration and the intermediate DED maps from projectionNN
    input: self - instantiation of the KINNTREX calss
           k - reaction rate coeficients
           TPoints - an array of time points
           I - Intermediate DED maps
           Lambda - Relaxation rates
    returns: Sampled_Conc - concetration values at input time points
             calcE2 - recalculated time-dependent DED maps
             EigVal - relaxation rates calculated from CDE solution.
    """
    def SolveCDE(self, k: Tensor, TPoints: Tensor, I: Tensor, LambdaIn: Tensor=None):
        tau     = np.arange(np.log(TPoints[0]),np.log(TPoints[-1])+self.dTau/2, self.dTau)
        tT      = np.exp(tau)
        tSample = self.tPoints
        if self.c2logk:
            Conc_t, EigVal = utils.Analytic_ODE_Solv (torch.exp(k), tT, self.initCond*self.initScale,
                                                      self.Model_Mat, LambdaIn)
        else:
            Conc_t, EigVal = utils.Analytic_ODE_Solv (k, tT, self.initCond*self.initScale, self.Model_Mat,
                                                      LambdaIn)
            
        Conc_t          = self.relu1(Conc_t)
        Sampled_Conc, _ = utils.getConc(tSample, tT, Conc_t)
        if self.incDark:
            calcE2 = torch.matmul(I, Sampled_Conc)
        else:
            calcE2 = torch.matmul(I, Sampled_Conc[:-1,:])
            
        return Sampled_Conc, calcE2, EigVal
    
    """
    The forward propagation method used for the propagation through KINNTREX within a single
    iteration.
    input: self - instantiation of the KINNTREX calss
           LSV - left singular vector values (only significant ones)
           TPoints - an array of time points.
    returns: Sampled_Conc - Concentration profiles at selected time points
             I - intermediate DED maps
             calcE1 - time-dependent DED maps calculated at projectionNN
             calcE2 - time-dependent DED maps calculated after solving CDE governing the
                      kinetics of the investigated molecule
             k - reaction rate coefficients
             Lambda_fromC - relaxation rates calculated from concentration
             Lambda_fromK - relaxation rates calculated from reaction rate coefficients
    """
    def forward(self, LSV: Tensor, TPoints: Tensor):
        # Projection Neural Network calculating time depenednt concentration values and DED maps 
        # for the intermediates as well as recalculating the time depenednt DED map. 
        # The input is the significant right singular vectors
        I, calcE1 = self.Projection_NN(LSV)
        
        # Conversion NN and Diagonal NN
        # Converting the concentrations into reaction rate coefficients (RRCs) or relaxation rates.
        # The convertion of RRCs to relaxation rates can be done either by usig NN or by diagonalizing 
        # the rate coefficient matrix K
        Lambda_fromC = None
        Lambda_fromK = None
        for Block in self.NN_Block_List:
            if Block == 'c2lambda':                     # conversion of cocentration to relaxation rates using
                Lambda_fromC = self.C2Lambda_NN()       # Neural Networks.
            elif Block == 'c2k':                        # conversion of cocentration to RRCs using
                k = self.C2k_NN()                       # Neural Networks.
            elif Block == 'lambda2k':                   # conversion of relaxation rates to RRCs using
                k = self.Lambda2k_NN(Lambda_fromC)      # Neural Networks.
                Lambda_fromC, _ = Lambda_fromC.sort(descending=True, stable=True)
            elif Block == 'k2lambda':                   # conversion of RRCs to relaxation rates using 
                Lambda_fromK = self.k2Lambda_NN(k) # Neural Networks.
            elif Block == 'Diag_k2lambda':              # conversion of RRCs to relaxation rates using
                Lambda_fromK = self.Diag_k2Lambda_NN(k) # matrix diagonalization.
            else:
                print('NN Block is not defined, blockis ignored')
        
        # Solving the coupled differential equations
        if self.B_Values is None:
            Sampled_Conc, calcE2, Lambda_fromDiag = self.SolveCDE(k, TPoints, I)
        elif 'Diag_k2lambda' in self.NN_Block_List or 'k2lambda' in self.NN_Block_List:
            Sampled_Conc, calcE2, Lambda_fromDiag = self.SolveCDE(k, TPoints, I)
        elif len(self.NN_Block_List) == 1 and self.NN_Block_List[0] == 'c2k':
            Sampled_Conc, calcE2, Lambda_fromDiag = self.SolveCDE(k, TPoints, I, -self.B_Values)
        else:
            Sampled_Conc, calcE2, _ = self.SolveCDE(k, TPoints, I, -self.B_Values)
            
        if Lambda_fromK is None and len(self.NN_Block_List) == 1 and self.NN_Block_List[0] == 'c2k':
            Lambda_fromK = Lambda_fromDiag

        return Sampled_Conc, I, calcE1, calcE2, k, Lambda_fromC, Lambda_fromK

    """
    This method is used to run KINNTREX iteratively while checking the quality of the results with 
    loss function and backpropagation. 
    input: self - instantiation of the KINNTREX calss
           Input - measured/simulated DED maps.
           Optimizer - The optimization method, usualy using 
    returns: Lambda - Relaxation rates.
    """
    def Training(self, Input, Optimizer=None):
        
        npLoss   = [] # array includes the loss values for selected iterations
        npLoss_e = [] # array includes the loss_e values for selected iterations
        npLoss_i = [] # array includes the loss_i values for selected iterations
        npLoss_B = [] # array includes the loss_B values for selected iterations
        npLoss_k = [] # array includes the loss_k values for selected iterations
        npLoss_c = [] # array includes the loss_c values for selected iterations
        nprCoef  = [] # array includes the k values for selected iterations
        npI      = [] # array includes the I arrays for selected iterations
        npIniSc  = [] # array includes the 'is' values for selected iterations
        
        LowLoss_Count = 0 # start counting sequential minimum loss values
        prevLoss = self.LossToll + 1.0 #initiating revLoss. prevLoss is the variable holding the previous interation loss value
        endTrain = False # a boolian variable holding True for stopping training process and False for continueing the training proces.
        
        # calculates the number of repeated minimum loss value required in case precentage of minimum loss value required is given
        if self.Percent_Repeat_LossToll is not None:
            Repeat_LossToll = round(self.num_epochs*self.Percent_Repeat_LossToll/100)
        else:
            Repeat_LossToll = self.Repeat_LossToll
        
        # selecting optimization function for backpropagation.
        if Optimizer is None:
            Optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

        if self.Input is None:
            gtE = Input
        else:
            gtE = self.Input
        
        gtE        = torch.from_numpy(gtE).float() # ground truths time-dependent DED maps
        LSV, _, _  = torch.svd(gtE) # ground truth left singular vectors
        logTPoints = np.log(self.tPoints.detach().numpy())
        dTauM      = np.mean(logTPoints[1:] - logTPoints[:-1])
        TStamp     = np.insert(self.tPoints, 0, np.exp(np.log(self.tPoints[0])-dTauM))
 
        # runs the forward propagation and backpropagation iteratively.
        for Iteration in range(self.num_epochs):
            if endTrain:
                break
                
            # Forward pass
            Conc, self.Interm, calcE1, calcE2, k, B_fromC, B_fromK = self.forward(LSV[:,0:self.nInterms], TStamp)
            # Loss value calculation
            loss, loss_e, loss_i, loss_B, loss_k, loss_c, loss_is = self.PINN_Loss(calcE1, calcE2, gtE, Conc, k, B_fromC, B_fromK)
                
            # checks if loss value is equal or lower than minum loss value selected and if the number of sequantial minimum loss values reached 
            if loss <= torch.tensor(self.LossToll) and LowLoss_Count == Repeat_LossToll:
                endTrain = True
            
            # counts the number of sequential minimum loss values
            if loss <= torch.tensor(self.LossToll) and prevLoss <= torch.tensor(self.LossToll):
                LowLoss_Count += 1
            elif loss <= torch.tensor(self.LossToll):
                LowLoss_Count = 1
            else:
                LowLoss_Count = 0
            prevLoss = loss
                
            # Backward and optimize
            Optimizer.zero_grad()
            loss.backward()
            Optimizer.step()
            
            # saving loss, k, I and 'is' values to arrays for recording purposes.
            if (Iteration+1) % 100 == 0:
                print (f'Epoch [{Iteration+1}/{self.num_epochs}], Loss: {loss.item():.7f}, Loss_E: {loss_e.item():.7f}, Loss_C: {loss_c.item():.7f}, Loss_K: {loss_k.item():.7f}') 
                npLoss.append(loss.item())
                npLoss_e.append(loss_e.item())
                npLoss_i.append(loss_i.item())
                npLoss_B.append(loss_B.item())
                npLoss_k.append(loss_k.item())
                npLoss_c.append(loss_c.item())
                npIniSc.append(self.initScale.item())
                nprCoef.append(k.detach().numpy().copy())
                npI.append(self.Interm.detach().numpy().copy())
                
            # Printing out loss and k values for monitoring purposes.
            if (Iteration+1) % 1000 == 0:
                Sub1 = '\N{subscript one}: '
                Sub2 = '\N{subscript two}: '
                Sub3 = '\N{subscript three}: '
                Sub4 = '\N{subscript four}: '
                Sub5 = '\N{subscript five}: '
                SubM = 'k\N{subscript minus}'
                print ('k' +\
                       Sub1 + f' {k[0]:.2f}, k' + Sub2 + f' {k[1]:.2f}, k' + Sub3 + f' {k[2]:.2f}, k' +\
                       Sub4 + f'{ k[3]:.2f}, k' + Sub5 + f' {k[4]:.2f}, \n' + SubM + Sub1 + f'{k[5]:.2f}, ' +\
                       SubM + Sub2 + f'{k[6]:.2f}, ' + SubM + Sub3 + f'{k[7]:.2f}, ' +\
                       SubM + Sub4 + f'{k[8]:.2f}, ' + SubM + Sub5 + f'{k[9]:.2f}')
                if self.B_Values is not None:
                    print ('\N{GREEK SMALL LETTER LAMDA}m' +\
                           Sub1 + f'{self.B_Values[0]:.4f}, \N{GREEK SMALL LETTER LAMDA}m' +\
                           Sub2 + f'{self.B_Values[1]:.4f}, \N{GREEK SMALL LETTER LAMDA}m' +\
                           Sub3 + f'{self.B_Values[2]:.4f}, \N{GREEK SMALL LETTER LAMDA}m' +\
                           Sub4 + f'{self.B_Values[3]:.4f}')
                    if B_fromK is None:
                        print ('\N{GREEK SMALL LETTER LAMDA}' +\
                               Sub1 + f'{B_fromC.real[0]:.4f}, \N{GREEK SMALL LETTER LAMDA}' +\
                               Sub2 + f'{B_fromC.real[1]:.4f}, \N{GREEK SMALL LETTER LAMDA}' +\
                               Sub3 + f'{B_fromC.real[2]:.4f}, \N{GREEK SMALL LETTER LAMDA}' +\
                               Sub4 + f'{B_fromC.real[3]:.4f}')
                    else:
                        print ('\N{GREEK SMALL LETTER LAMDA}' +\
                               Sub1 + f'{B_fromK.real[0]:.4f}, \N{GREEK SMALL LETTER LAMDA}' +\
                               Sub2 + f'{B_fromK.real[1]:.4f}, \N{GREEK SMALL LETTER LAMDA}' +\
                               Sub3 + f'{B_fromK.real[2]:.4f}, \N{GREEK SMALL LETTER LAMDA}' +\
                               Sub4 + f'{B_fromK.real[3]:.4f}')
                print(f'Init. Scale: {self.initScale.data}')

                
        Last_OutData    = {'I': self.Interm, 'Rates': k, 'InitScale': self.initScale}
        Sampled_OutData = {'Loss':   npLoss,   'Loss_e': npLoss_e, 'Loss_i': npLoss_i, 'Loss_B': npLoss_B,
                           'Loss_k': npLoss_k, 'Loss_c': npLoss_c, 'Rate':   nprCoef,  'I':      npI,
                           'InitScale': npIniSc}
        print(f'NN Architecture {self.NN_Block_List}')
        return Last_OutData, Sampled_OutData