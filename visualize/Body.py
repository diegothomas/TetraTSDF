
import numpy as np
import math
from numpy import linalg as LA
import imp
import time

GPU = imp.load_source('GPUManager', './GPUManager.py')
TSDFtk = imp.load_source('TSDFtk', './TSDFManager.py')
My_MT = imp.load_source('MarchingTetrahedra', './MarchingTetrahedra.py')
Warp = imp.load_source('warpVolume', './warpVolume.py')

class Body():
    def __init__(self, TSDF,Weight):

        self.TSDF = TSDF
        self.Weight = Weight

    def updateTables(self,updatedTSDF,updatedWeight):

        self.TSDF = updatedTSDF
        self.Weight = updatedWeight
    
    def getTSDF(self):
        return self.TSDF
