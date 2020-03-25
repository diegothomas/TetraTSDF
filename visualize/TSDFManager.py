#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 17:16:49 2017

@author: diegothomas
"""

import imp
import numpy as np
from numpy import linalg as LA
import pyopencl as cl
from array import array
import open3d as o3d

#RGBD = imp.load_source('RGBD', './lib/RGBD.py')
GPU = imp.load_source('GPUManager', './GPUManager.py')
KernelsOpenCL = imp.load_source('KernelsOpenCL', './KernelsOpenCL.py')
My_MT = imp.load_source('MarchingTetrahedra', './MarchingTetrahedra.py')
#General = imp.load_source('General', './lib/General.py')

savepath = "output/debug"

def sign(x):
    if (x < 0):
        return -1.0
    return 1.0


mf = cl.mem_flags

def depth2ply(img, out_path, intrinsic_mat, extrinsic_mat=np.eye(4, dtype=np.float32)):

    # Depth img to ply

    scale = 1000.0 # Kinect v2

    fx = intrinsic_mat[0,0]
    fy = intrinsic_mat[1,1]
    cx = intrinsic_mat[0,2]
    cy = intrinsic_mat[1,2]

    row = img.shape[0]
    col = img.shape[1]
    x = np.array([np.arange(col, dtype=np.float32) for i in range(row)])
    y = np.array([np.full(col, i, dtype=np.float32) for i in range(row)])
    z = img/scale
    x = ((x - cx)*z/fx).flatten()
    y = ((y - cy)*z/fy).flatten()
    z = z.flatten()
    V = np.vstack([np.vstack([x, y]), z]).T
    valid = np.any(V>0, axis=1)
    V = V[valid]
    V_after = np.matmul(extrinsic_mat, np.hstack([V, np.ones([len(V), 1])]).T)[0:3].T
    # V_after = V_after*np.array([1., -1., 1.])
    saveply(out_path, V_after)

def saveply(path, V):
    with open(path, "w") as f:
        # Write headers
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment ply file \n")
        f.write("element vertex %d \n" %(V.shape[0]))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for v in V:
            f.write("%f %f %f\n" %(v[0], v[1], v[2]))
    
class TSDFManager():
    """
    Manager Truncated Signed Distance Function.
    TSDFManager = TSDFtk.TSDFManager(c_mesh,depth,voxelArray,vertexArray,intrinsic,GPUManager,PoseBP,VoxSize,VertexCnt)

    """
    def createGPUArray(self):
        result = np.zeros((self.Size[2]*4,3),dtype = np.float)
        result = result.ravel()
        for voxid in range(0,self.Size[2]):
            for vertid in range(0,4):
                idx = 4 * voxid +vertid
                result[idx*3+0] = self.VertexArray[(self.VoxelArray[idx])*3+0] 
                result[idx*3+1] = self.VertexArray[(self.VoxelArray[idx])*3+1] 
                result[idx*3+2] = self.VertexArray[(self.VoxelArray[idx])*3+2]
        return result

    def __init__(self, Size, voxelvertexArray,voxelArray,vertexArray, GPUManager,Tg, VoxCnt, VertexCnt,nu,StartWeight,StartTSDF, maximum_mode=True): #TODO bunun son argumani bos 
        """
        Constructor
        :param Image: RGBD image to compare
        :param GPUManager: GPU environment for GPU computation
        :param VoxCnt: Kac voxel var 
        :param intrinsic : intrinsic matrixi np.aray
        :param Tg: transform from the local coordinate to global coordinate identitiy matrix
        :param VertexCnt: Kac vertex var(her point x y z coord olusuyor bu x3 bilgimiz olacak) 
        """
        ### Onizuka added#########
        convVal = 32767.0
        num_vertices = len(vertexArray)
        self.maximum_mode = maximum_mode
        if maximum_mode: self.TSDF_vertices = np.full(num_vertices, -convVal, dtype=np.float32)
        else: self.TSDF_vertices = np.full(num_vertices, convVal, dtype=np.float32)
        # self.TSDF_vertices = np.full(num_vertices, 0, dtype=np.float32)
        self.Weights_vertices = np.zeros(num_vertices, dtype=np.float32)
        #########################

        print("Image shape:")
        self.depth = np.zeros((424,512),dtype = np.int16)
        self.nu = nu
        #np.savetxt('Image.txt', self.Image, delimiter=',')
        #dimensions bunlari kullanma 
        self.Size = Size
        print("==========SIZE=========")
        print(self.Size)
        print("=======================")
        #We dont need the center points 
        #self.c_x = Size[0]/2
        #self.c_y = Size[1]/2
        #self.c_z = Size[2]/2
    
        self.VoxCnt = VoxCnt
        self.VertexCnt = VertexCnt 
        self.VoxelArray = voxelArray #int
        self.VoxelArray = self.VoxelArray.ravel() #int
        #self.VertexArray = np.zeros((56559*3),)

        #self.VertexArray = copy.deepcopy(vertexArray) #float
        self.VertexArray = vertexArray #floatnp.load('vertexArray.npy')#
        # self.VertexArray =  self.VertexArray.ravel()
        #np.savetxt('vertexdogrumu1.txt', self.VertexArray, delimiter=',')
        #np.savetxt('vertexdogrumu3.txt', vertexArray, delimiter=',')
        #np.savetxt('voxelArraydogrumu1.txt', voxelArray, delimiter=',')
        #np.savetxt('voxelArraydogrumu2  .txt', self.VoxelArray, delimiter=',')
        #self.TSDFtable = TSDFtable
        
        #self.voxeldogrumu = np.zeros((voxelArray.shape),dtype=np.int32)
        #self.onesones = np.ones((56559*3),dtype=np.float32)
        # resolution, do we need it?
        #VoxSize = 0.005
        #self.dim_x = 1.0/VoxSize
        #self.dim_y = 1.0/VoxSize
        #self.dim_z = 1.0/VoxSize
        # put dimensions and resolution in one vector
        #self.res = np.array([self.c_x, self.dim_x, self.c_y, self.dim_y, self.c_z, self.dim_z], dtype = np.float32)

        self.GPUManager = GPUManager
        #self.Size_Volume = cl.Buffer(self.GPUManager.context, mf.READ_ONLY | mf.COPY_HOST_PTR, \
        #                       hostbuf = np.array([self.Size[0], self.Size[1], self.Size[2]], dtype = np.float64))
        #self.Size_Volume = cl.Buffer(self.GPUManager.context, mf.READ_ONLY | mf.COPY_HOST_PTR, \
        #                       hostbuf = np.array([self.Size[1], self.Size[2]], dtype = np.float64))
        #self.TSDF = np.zeros((self.Size[2],self.Size[1]), dtype=np.float64)
        #GPU ICIN SHORT INT OLMALI ASAGIDA
        #self.TSDF = np.zeros((self.Size[2],self.Size[1]), dtype=np.int16)
        self.TSDF = StartTSDF.flatten()
        self.TSDFGPU =cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.TSDF.nbytes)

        #distance values to checl 
        self.Dist = np.zeros((self.Size[2],self.Size[1]),dtype=np.int16)
        self.Dist = self.Dist.ravel()
        self.DistGPU =cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.Dist.nbytes)
        #self.Weight = np.zeros(self.Size, dtype=np.int16)
        #self.WeightGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.Weight.nbytes)

        N = VertexCnt*3

        #self.VertexGPU =cl.Buffer(self.GPUManager.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf =  self.VertexArray)



        #self.VoxelGPU =cl.Buffer(self.GPUManager.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = self.VoxelArray)
        #self.TSDFtableGPU =cl.Buffer(self.GPUManager.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = TSDFtable) #TODO bu bosta

        #self.Param = cl.Buffer(self.GPUManager.context, mf.READ_ONLY | mf.COPY_HOST_PTR, \
        #                       hostbuf = self.res)
        #self.VertexGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, vertexArray.nbytes)
        self.DepthGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.depth.nbytes)
        #self.DepthGPU =cl.Buffer(self.GPUManager.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf =  self.Image)
        
        # self.Calib_GPU = cl.Buffer(self.GPUManager.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = intrinsic)
        
        # self.Pose = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
        # self.Pose_GPU = cl.Buffer(self.GPUManager.context, mf.READ_ONLY, self.Pose.nbytes)
        # cl.enqueue_write_buffer(self.GPUManager.queue, self.Pose_GPU, Tg).wait()

        #weightler 
        #self.Weight = np.ones((self.Size[1],self.Size[2]), dtype=np.int16)
        self.Weight = StartWeight
        self.Weight = self.Weight.ravel()
        self.WeightGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.Weight.nbytes)

        #self.sortedVertex = np.zeros((self.Size[2]*4,3),dtype = np.float)
        #self.sortedVertex = self.createGPUArray()
        #np.save('sortedVertex.npy', self.sortedVertex)
        #self.sortedVertex = np.load('sortedVertex.npy')
        #print("self.sortedVertex::::::")
        #print(self.sortedVertex.shape)
        #self.sortedVertex  = self.sortedVertex.ravel()
        #print(self.sortedVertex.shape)
        #self.sortedVertexGPU =cl.Buffer(self.GPUManager.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = self.sortedVertex)
        #self.sortedsize = (self.Size[2])*4
        #print("self.sortedsize")
        #print(self.sortedsize)
        self.imagevals = np.zeros((self.Size[2],self.Size[1]), dtype=np.int32)
        self.imagevals = self.imagevals.ravel()
        # calculate the corner weight
        #self.planeF = cl.Buffer(self.GPUManager.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=planeF)
        #tempDQ = np.zeros((2,4), dtype=np.float32)
        #self.boneDQGPU = cl.Buffer(self.GPUManager.context, mf.READ_ONLY, tempDQ.nbytes)
        #self.jointDQGPU = cl.Buffer(self.GPUManager.context, mf.READ_ONLY, tempDQ.nbytes)
        #For pixels 
        
        self.pixelsx = np.zeros((self.Size[2],self.Size[1]), dtype=np.int32)
        self.pixelsy = np.zeros((self.Size[2],self.Size[1]), dtype=np.int32)
        self.indices = np.zeros((self.Size[2],self.Size[1]), dtype=np.int32)
        
        self.pointsx = np.zeros((self.Size[2],self.Size[1]), dtype=np.float32)
        self.pointsy = np.zeros((self.Size[2],self.Size[1]), dtype=np.float32)
        self.pointsz = np.zeros((self.Size[2],self.Size[1]), dtype=np.float32)

        self.pixelsx = self.pixelsx.ravel()
        self.pixelsy = self.pixelsy.ravel()
        self.indices = self.indices.ravel()
        
        self.pointsx = self.pointsx.ravel()
        self.pointsy = self.pointsy.ravel()
        self.pointsz = self.pointsz.ravel()
        

        #Kac tane pixel degeri kullaniyoruz gormek icin 
        self.pixelstotal = np.zeros((self.Size[2],self.Size[1]), dtype=np.int32)
        self.pixelstotal = self.pixelstotal.ravel()
        #self.pixelsxGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.pixelsx.nbytes)
        #self.pixelsyGPU = cl.Buffer(self.GPUManager.context, mf.READ_WRITE, self.pixelsy.nbytes)

        self.voxelvertexArray = voxelvertexArray
        self.voxelvertexArray = self.voxelvertexArray.ravel()
        self.voxelvertexArrayGPU =cl.Buffer(self.GPUManager.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = self.voxelvertexArray)



#######
##GPU code
####

    # Fuse on the GPU
    #def FuseRGBD_GPU(self, TSDF,Weight,depth_image,depth_shape,nu):
    def FuseRGBD_GPU(self,depth_image,depth_intrinsic, cam_pose, nu):
        """
        Update the TSDF volume with Image
        :param Image: RGBD image to update to its surfaces
        """
        # initialize buffers
        #oneweights = np.ones((self.Size[2],self.Size[1]), dtype=np.int16)
        self.nu = nu
        #self.TSDF = TSDF
        #self.Weight = Weight
        cl.enqueue_write_buffer(self.GPUManager.queue, self.DepthGPU, depth_image)
        #now oneweights ise 1leri attim ilerde buu skinning weightleri ile dene 
        #cl.enqueue_write_buffer(self.GPUManager.queue, self.WeightGPU, oneweights).wait()
        
        #cl.enqueue_write_buffer(self.GPUManager.queue, self.VertexGPU, self.VertexArray).wait()
        #cl.enqueue_write_buffer(self.GPUManager.queue, self.boneDQGPU, boneDQ)
        #cl.enqueue_write_buffer(self.GPUManager.queue, self.jointDQGPU, jointDQ)
        
        #print(np.int32(depth_image.shape[1])) simdi ravel gonderiyorum bu yuzden boyle 
        # fuse data of the RGBD imnage with the TSDF volume 3D model
        print("==========DEPTH=fuseRGBD========")
        print(depth_image.shape[0])
        print(depth_image.shape[1])
        print(self.Size[1])
        print(self.Size[2])
        print("=======================")
        
        
        
        #self.GPUManager.programs['FuseTSDF'].FuseTSDF(self.GPUManager.queue, (self.Size[1], self.Size[2]) , None, \
        #                        self.TSDFGPU, self.WeightGPU,self.DistGPU,self.DepthGPU, self.Pose_GPU, \
        #                        self.VoxelGPU,self.VertexGPU, \
        #                        self.Calib_GPU, np.int32(depth_shape[0]), np.int32(depth_shape[1]),\
        #                        )

        '''
        self.GPUManager.programs['FuseSortedTSDF'].FuseSortedTSDF(self.GPUManager.queue, (self.Size[2], self.Size[1]) , None, \
                        self.TSDFGPU, self.WeightGPU,self.DistGPU,self.DepthGPU, \
                        self.voxelvertexArrayGPU, np.int32(self.nu), \
                        self.Calib_GPU, np.int32(depth_shape[0]), np.int32(depth_shape[1])\
                        )
        
        #GPU for Sphere Equation
        #self.GPUManager.programs['FuseEqTSDF'].FuseEqTSDF(self.GPUManager.queue, (self.Size[1], self.Size[2]), None, \
        #                        self.TSDFGPU, self.TSDFtableGPU,self.Size_Volume, self.Pose_GPU, \
        #                        self.VoxelGPU,self.VertexGPU, \
        #                        np.int32(depth_shape[0]), np.int32(depth_shape[1]))


        # update CPU array. Read the buffer to write in the CPU array.
        cl.enqueue_read_buffer(self.GPUManager.queue, self.TSDFGPU, self.TSDF).wait()
        cl.enqueue_read_buffer(self.GPUManager.queue, self.DistGPU, self.Dist).wait()
        cl.enqueue_read_buffer(self.GPUManager.queue, self.WeightGPU, self.Weight).wait()
        #cl.enqueue_read_buffer(self.GPUManager.queue, self.DepthGPU, self.depth).wait()
        #cl.enqueue_read_buffer(self.GPUManager.queue, self.VoxelGPU, self.voxeldogrumu).wait()
        '''
        
        
        #FOR CPU calculation
        # self.calculateTSDF(depth_image, depth_intrinsic, cam_pose)
        if(self.maximum_mode): self.calculateTSDF_absminmode(depth_image, depth_intrinsic, cam_pose)
        else: self.calculateTSDF_ver2(depth_image, depth_intrinsic, cam_pose)
        # np.savetxt(savepath + '/FuseRGBD_CGPU.txt', self.TSDF_vertices, delimiter=',')
        # np.savetxt(savepath + '/FuseRGBD_CDist.txt', self.Dist, delimiter=',')
        
        #from Sphere Equation
        #self.TSDFeq()
        
        # np.savetxt(savepath + '/FuseRGBD_CGPU.txt', self.TSDF, delimiter=',')
        # np.savetxt(savepath + '/FuseRGBD_CDist.txt', self.Dist, delimiter=',')
        #np.savetxt('FuseRGBD_WeightGG.txt', self.Weight, delimiter=',')
        #np.savetxt('FuseRGBD_cokyanlis.txt', self.depth, delimiter=',')
    
        
        # TEST if TSDF contains NaN
        TSDFNaN = np.count_nonzero(np.isnan(self.TSDF_vertices))
        print ("TSDFNaN : {0:d}".format(TSDFNaN))

        return self.TSDF_vertices,self.Weights_vertices
        # return self.TSDF,self.Weight
        
        

#####
#End GPU code
#####

    def calculateTSDF(self, depth_image, depth_intrinsic, cam_pose):
        depth_image = depth_image.flatten()
        convVal = 32767.0
        nu = self.nu#8711.1#8662.0
        pt = np.zeros(3)
        pix = np.zeros(2, dtype=np.int32)
        dist = 0.
        text_filepix = open(savepath + "/PixelsInCubeAfter.txt", "w")
        text_filepon = open(savepath + "/PointsInCubeAfter.txt", "w")
        for voxid in range(0,self.Size[2]):
        #for voxid in range(0,36):
            for vertid in range(0,4):
                idx = 4 * voxid +vertid
                pt = np.dot(cam_pose, np.hstack([self.VertexArray[self.VoxelArray[idx]], 1]))[0:3] 

                # pix = np.dot(depth_intrinsic, pt)[0:2].astype(np.int32)
                pix[0] = int(pt[0]/pt[2]*depth_intrinsic[0,0] + depth_intrinsic[0,2] )
                pix[1] = int(pt[1]/pt[2]*depth_intrinsic[1,1] + depth_intrinsic[1,2] )
                
                text_filepon.write("'%s', '%s', '%s', '%s'" %(pt[0],pt[1],pt[2],idx)+ "\n")
                text_filepix.write("'%s', '%s', '%s'" %(pix[0],pix[1],idx)+ "\n")
                if (pix[0] < 0 or pix[0] > 511 or pix[1] < 0 or pix[1] > 423):
                
                    self.TSDF[idx] = int(convVal)
                    #self.Dist[idx] = 0
                
                else:
                    tmpimg[pix[1], pix[0]] = 255

                    index = int(pix[0] + 512*pix[1])
                    self.pixelstotal[idx] = index
                    self.imagevals[idx] = depth_image[index]

                    if(depth_image[index]==0):
                        self.TSDF[idx] = int(convVal)
                        #self.Dist[idx] = 0
                    
                    else:
                        
                        #buna dikat et
                        ptz = float(pt[2])# * nu  
                        
                        #dist = -(ptz - self.Image[index])# / convVal
                        dist = -(ptz - float(depth_image[index])/1000.)
                        
                        #if (dist <4000) and (dist > -4000 ):
                        #if dist > 0.0 :
                        #    dist = dist / self.Image[index]
                        #else : 
                        #    dist = dist / (convVal - self.Image[index]) 
                        tdist = min(1.0, max(-1.0, dist))
                        self.Dist[idx] = tdist
                        #Running Average

                        prev_tsdf = (float(self.TSDF[idx])/convVal)
                        prev_weight = (float)(self.Weight[idx])
                        w = 1.0#self.Weight[idx]
                        #self.Dist[idx] =idx
                        #if (dist<(float)(TSDF[idx])/convVal) w=0.1
                        #3if (dist < 1. and dist > -1.):
                        #    print("Bakin bu bir dramdir ")
                        #    print(((prev_tsdf*prev_weight+tdist*w)/(prev_weight+w))*convVal)
                        
                        #For weights  
                        if tdist >-1 and tdist < 1:
                            self.TSDF[idx] = int(round(((prev_tsdf*prev_weight+tdist*w)/(prev_weight+w))*convVal))
                         #outside,inside in out kavramlarina gore bu degisebilir aslinda 
                            self.Weight[idx] = self.Weight[idx]+1
                        #self.TSDF[idx] = round(tdist * convVal)
                        
                        self.indices[idx] = ptz
                        self.pixelsx[idx] = pix[0]
                        self.pixelsy[idx] = pix[1]
                        self.pointsx[idx] = pt[0]
                        self.pointsy[idx] = pt[1]
                        self.pointsz[idx] = pt[2]
                        #else :
                        #    self.TSDF[idx] = 1.0

        #text_filepon.close()
        #text_filepix.close() 
        #mlist = list(set(formatted))
        #print(mlist[-1] )
        #slist=sorted(mlist)
        #print(slist[3] )
        #print(self.Dist.max())
        #print(np.absolute(self.Dist).min())
        
        np.savetxt(savepath + '/FuseRGBD_dogruindexler.txt',self.imagevals, delimiter=',')
        return self.TSDF,self.Weight
        print("=======================")


    def calculateTSDF_ver2(self, depth_image, depth_intrinsic, cam_pose):

        w = 1.0
        convVal = 32767.0
        rows = depth_image.shape[0]
        cols = depth_image.shape[1]
        dscale = 1000 # Kinect v2 dimg [mm] -> [m]

        # Rotate and translate vertices and Project to 2d
        # VertexArray_local = np.matmul(cam_pose, np.hstack([self.VertexArray, np.ones([len(self.VertexArray), 1])]).T)[0:3].T
        VertexArray_local = np.matmul(cam_pose[:3, :3].T, (self.VertexArray - cam_pose[:3,3].T).T).T
        VertexArray_2dproj = ((VertexArray_local[:,0:2]/VertexArray_local[:,2].reshape(-1, 1))*np.array([depth_intrinsic[0,0], depth_intrinsic[1,1]]) + np.array([depth_intrinsic[0,2], depth_intrinsic[1,2]])).astype(np.int32)
        # camposeT = np.hstack([cam_pose[:3, :3].T, -np.dot(cam_pose[:3, :3].T, np.array([cam_pose[:3, 3]]).T)])
        # camposeT = np.vstack([camposeT, np.array([0,0,0,1])])
        # depth2ply(depth_image, "./tools/pcfromdimg.ply", depth_intrinsic, camposeT)
        # depth2ply(depth_image, "./tools/pcfromdimg.ply", depth_intrinsic, cam_pose)
        # depth2ply(depth_image, "./tools/pcfromdimg.ply", depth_intrinsic, np.linalg.inv(cam_pose))

        # Calculate Distance to surface from each vertices
        conditions = np.array([0<=VertexArray_2dproj[:,1], VertexArray_2dproj[:,1]<rows, 0<=VertexArray_2dproj[:,0], VertexArray_2dproj[:,0]<cols])
        xlist = np.where(conditions.all(axis=0), VertexArray_2dproj[:,0], 0.0)
        ylist = np.where(conditions.all(axis=0), VertexArray_2dproj[:,1], 0.0)
        DepthValues = depth_image[ylist.astype(np.int32), xlist.astype(np.int32)].astype(np.float32)/dscale
        # DepthValues = np.where(conditions.all(axis=0), depth_image[VertexArray_2dproj[:,1], VertexArray_2dproj[:,0]], 0.0)/dscale     #Doesn't work correctly (why?)
        DistValues = (DepthValues - VertexArray_local[:,2])/self.nu
        TruncatedDist = np.minimum(1.0, np.where(DistValues<-1.0, 1.0, DistValues)) # If not-nu < distance < nu : tsdf = 1 (outside)
        ValidVerts = np.where(np.array([-1.0<TruncatedDist, TruncatedDist<1.0]).all(axis=0), True, False)     # -thres<TruncatedDist<thres -> Update
        
        #self.saveply("./input/raw/TSDF/for.ply", VertexArray_local[ValidVerts]) #for debug
        
        # Compute TSDF and update Weights (for only ValidVerts)
        prev_tsdf = (self.TSDF_vertices / convVal).astype(np.float32)
        self.TSDF_vertices = np.where(ValidVerts, (np.round(((prev_tsdf * self.Weights_vertices + TruncatedDist)/(self.Weights_vertices + w))*convVal)).astype(np.int16), self.TSDF_vertices)
        self.Weights_vertices += ValidVerts

        return self.TSDF_vertices, self.Weights_vertices

    def calculateTSDF_maxmode(self, depth_image, depth_intrinsic, cam_pose):

        w = 1.0
        convVal = 32767.0
        rows = depth_image.shape[0]
        cols = depth_image.shape[1]
        dscale = 1000.0 # Kinect v2 dimg [mm] -> [m]

        # Rotate and translate vertices and Project to 2d
        # VertexArray_local = np.matmul(cam_pose, np.hstack([self.VertexArray, np.ones([len(self.VertexArray), 1])]).T)[0:3].T
        VertexArray_local = np.matmul(cam_pose[:3, :3].T, (self.VertexArray - cam_pose[:3,3].T).T).T
        VertexArray_2dproj = ((VertexArray_local[:,0:2]/VertexArray_local[:,2].reshape(-1, 1))*np.array([depth_intrinsic[0,0], depth_intrinsic[1,1]]) + np.array([depth_intrinsic[0,2], depth_intrinsic[1,2]])).astype(np.int32)
        
        # Calculate Distance to surface from each vertices
        conditions = np.array([0<=VertexArray_2dproj[:,1], VertexArray_2dproj[:,1]<rows, 0<=VertexArray_2dproj[:,0], VertexArray_2dproj[:,0]<cols])
        DepthValues = np.where(conditions.all(axis=0), depth_image[VertexArray_2dproj[:,1], VertexArray_2dproj[:,0]], 0.0)/dscale
        DistValues = (DepthValues - VertexArray_local[:,2])/self.nu
        TruncatedDist = np.minimum(1.0, np.maximum(-1.0, DistValues))
        # TruncatedDist = np.minimum(1.0, np.where(DistValues<-1.0, 1.0, DistValues)) # If not-nu < distance < nu : tsdf = 1 (outside)
        # ValidVerts = np.where(np.array([-1.0<TruncatedDist, TruncatedDist<1.0]).all(axis=0), True, False)     # -thres<TruncatedDist<thres -> Update
        self.saveply("./test.ply", VertexArray_local)
        
        # Compute TSDF and update Weights (for only ValidVerts)
        prev_tsdf = (self.TSDF_vertices / convVal).astype(np.float32)
        self.TSDF_vertices = (np.maximum(prev_tsdf, TruncatedDist)*convVal).astype(np.int16)
        # self.TSDF_vertices = np.where(ValidVerts, (np.round(((prev_tsdf * self.Weights_vertices + TruncatedDist)/(self.Weights_vertices + w))*convVal)).astype(np.int16), self.TSDF_vertices)
        # self.Weights_vertices += ValidVerts

        return self.TSDF_vertices, self.Weights_vertices

    def calculateTSDF_absminmode(self, depth_image, depth_intrinsic, cam_pose):

        w = 1.0
        convVal = 32767.0
        rows = depth_image.shape[0]
        cols = depth_image.shape[1]
        dscale = 1000.0 # Kinect v2 dimg [mm] -> [m]

        # Rotate and translate vertices and Project to 2d
        # VertexArray_local = np.matmul(cam_pose, np.hstack([self.VertexArray, np.ones([len(self.VertexArray), 1])]).T)[0:3].T
        VertexArray_local = np.matmul(cam_pose[:3, :3].T, (self.VertexArray - cam_pose[:3,3].T).T).T
        VertexArray_2dproj = ((VertexArray_local[:,0:2]/VertexArray_local[:,2].reshape(-1, 1))*np.array([depth_intrinsic[0,0], depth_intrinsic[1,1]]) + np.array([depth_intrinsic[0,2], depth_intrinsic[1,2]])).astype(np.int32)
        
        # Calculate Distance to surface from each vertices
        conditions = np.array([0<=VertexArray_2dproj[:,1], VertexArray_2dproj[:,1]<rows, 0<=VertexArray_2dproj[:,0], VertexArray_2dproj[:,0]<cols])
        DepthValues = np.where(conditions.all(axis=0), depth_image[VertexArray_2dproj[:,1], VertexArray_2dproj[:,0]], 0.0)/dscale
        DistValues = (DepthValues - VertexArray_local[:,2])/self.nu
        TruncatedDist = DistValues
        # TruncatedDist = np.minimum(1.0, np.maximum(-1.0, DistValues))
        # TruncatedDist = np.minimum(1.0, np.where(DistValues<-1.0, 1.0, DistValues)) # If not-nu < distance < nu : tsdf = 1 (outside)
        # ValidVerts = np.where(np.array([-1.0<TruncatedDist, TruncatedDist<1.0]).all(axis=0), True, False)     # -thres<TruncatedDist<thres -> Update
        self.saveply("./test.ply", VertexArray_local)
        
        # Compute TSDF and update Weights (for only ValidVerts)
        prev_tsdf = (self.TSDF_vertices / convVal).astype(np.float32)
        self.TSDF_vertices = (np.where(np.abs(prev_tsdf)<np.abs(TruncatedDist), prev_tsdf, TruncatedDist)*convVal).astype(np.int16)
        # self.TSDF_vertices = np.where(ValidVerts, (np.round(((prev_tsdf * self.Weights_vertices + TruncatedDist)/(self.Weights_vertices + w))*convVal)).astype(np.int16), self.TSDF_vertices)
        # self.Weights_vertices += ValidVerts

        return self.TSDF_vertices, self.Weights_vertices

    

    def TSDFeq(self):
        convVal = 32767.0
        pt = np.zeros(3)
        pix = np.zeros(2)

        for voxid in range(0,self.Size[2]):
        #for voxid in range(0,36):
            for vertid in range(0,4):
                idx = 4 * voxid +vertid
                pt[0] = self.VertexArray[(self.VoxelArray[idx])*3+0] 
                pt[1] = self.VertexArray[(self.VoxelArray[idx])*3+1] 
                pt[2] = self.VertexArray[(self.VoxelArray[idx])*3+2]

                dist = (pt[0]-1.5)*(pt[0]-1.5) + (pt[1]-1.5)*(pt[1]-1.5) + (pt[2]-1.5)*(pt[2]-1.5) 
                dist = pow(dist,0.5) 
                if dist < 0.4 :
                    self.TSDF[idx] = (dist - 0.4 )/ 0.4               
                elif dist == 0.4 :
                    self.TSDF[idx] = 0.0
                
                elif dist > 0.4:
                    self.TSDF[idx] = (dist - 0.4 )/ 0.44


