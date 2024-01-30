# Define Data Generating class
import numpy
import torch
import math
import matplotlib.pyplot as plt
import mAECell_Utils as utils
           
class FakeData:

    nTPoints: int
    tPoints : float
    nVoxels : int
    nInterms: int
    nTSeries: int
    Conc    : float
    Interms : float
    DataSet = []
    
    def __init__(self, nTPoints: int=None, batch_size: int=None, nVoxels: int=None, 
                 nInterms: int=None, nTSeries: int=None,
                 PlotAttr: dict={'Save_Plot'    : True,
                                 'nPlots'       : 3,
                                 'nxPanels'     : 2,
                                 'nyPanels'     : 2,
                                 'nPoints'      : 21,
                                 'yLimit'       : [0, 0.2],
                                 'PlotType'     : 'Concentration',
                                 'wNoise'       : True,
                                 'Sample_Number': 1,
                                 'Interval'     : 1}):
        self.nTPoints   = nTPoints
        self.batch_size = batch_size
        self.nVoxels    = nVoxels
        self.nInterms   = nInterms
        self.nTSeries   = nTSeries
        self.Conc       = numpy.zeros((nInterms + 1, nTPoints))

        for Key in PlotAttr:
            setattr(self, Key, PlotAttr[Key])

    def Load_eDensDiff(self, File_Name):
        loaded_array = numpy.loadtxt(File_Name)
        return loaded_array

    def Load_ConcCont(self, File_Name, TPoints = None, update_nInterms=True):
        if self.tPoints is None:
            if TPoints is not None:
                self.tPoints = TPoints
        loaded_array = numpy.loadtxt(File_Name)
        self.Conc = self.getConc(loaded_array[:, 0], loaded_array[:, 1:], update_nInterms)
        
        return loaded_array[:, 0], loaded_array[:, 1:]

    def Load_MeasTimStmp(self, File_Name: str = None, tPoints: numpy.double = None):
        if tPoints is None:
            if File_Name is None:
                File_Name = utils.fOpen(Title = 'Load Time Stamps')
            loaded_array = numpy.loadtxt(File_Name, usecols=0, dtype='str')
            nTPoints     = int(loaded_array[0])
            tPoints      = loaded_array[1:nTPoints + 1].astype(float)
        else:
            nTPoints = len(tPoints)
        self.nTPoints = nTPoints
        self.tPoints  = tPoints
        logTPoints    = numpy.log(self.tPoints)
        dTauM         = numpy.mean(logTPoints[1:] - logTPoints[:-1])
        return dTauM

    def getConc(self, cTS, cValue, update_nInterms=True):
        row, col = cValue.shape
        if update_nInterms:
            self.nInterms = col
        Conc = numpy.zeros((col, self.nTPoints))
        for ii in range(self.nTPoints):
            difference_array = numpy.absolute(cTS - self.tPoints[ii])
            indConcTS = difference_array.argmin()
            Conc[:, ii] = cValue[indConcTS, :]
        return Conc

    def Creat_DataSets(self, Files_Names, Index_Included):
        if round(numpy.sum(self.Conc), 0) == 0:
            print("No concentration data exist")
            return -1

        Conc = self.Conc
        if Index_Included:
            Loaded_Data  = self.Load_eDensDiff(Files_Names[0])
            aryFake_Data = Loaded_Data[:,1:]
            aryFake_Data = numpy.concatenate((aryFake_Data, 
                                              numpy.expand_dims(self.tPoints, axis=0)),
                                             axis=0)
        else:
            aryFake_Data = self.Load_eDensDiff(Files_Names[0])
            aryFake_Data = numpy.concatenate((aryFake_Data, 
                                              numpy.expand_dims(self.tPoints, axis=0)),
                                             axis=0)
        aryRow, _ = aryFake_Data.shape

        for ii in range(1, self.nTSeries):
            if Index_Included:
                Loaded_Data = self.Load_eDensDiff(Files_Names[ii])
                Fake_Data   = Loaded_Data[:,1:]
                Fake_Data   = numpy.concatenate((Fake_Data, 
                                                 numpy.expand_dims(self.tPoints, axis=0)),
                                                axis=0)
            else:
                Fake_Data = self.Load_eDensDiff(Files_Names[ii])
                Fake_Data = numpy.concatenate((Fake_Data, 
                                               numpy.expand_dims(self.tPoints, axis=0)),
                                              axis=0)
            row, _ = Fake_Data.shape
            if aryRow == row:
                Conc = numpy.hstack([Conc, self.Conc])
                aryFake_Data = numpy.hstack((aryFake_Data, Fake_Data))
        dataConc     = torch.tensor(Conc.conj())
        Fake_DataSet = torch.tensor(aryFake_Data.conj())
        DataSet      = torch.utils.data.TensorDataset(Fake_DataSet.T, dataConc.T)
        self.DataSet = torch.utils.data.DataLoader(DataSet, batch_size=self.batch_size, 
                                                   shuffle=False)

    def getSample(self, iSample):
        Data        = enumerate(self.DataSet)
        if iSample < 1:
            print("Sample number is too low")
            return -1
        else:
            for Step in range(iSample):
                Item = next(Data)
        Fake_Sample = Item[1][0]
        Conc        = Item[1][1]
        return Fake_Sample, Conc