from argparse import ArgumentParser
import os
from os.path import join, exists, abspath, dirname, basename, splitext
import glob
import pymesh 
import skimage
import cv2
import numpy as np
import numpy.linalg as LA
import Tkinter as tk
import tkFileDialog
import cPickle as pickle

###bunlar image gostermek icin ekliyorum
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imp
import time
from math import sqrt
import xml.etree.ElementTree as ET

###########################
import csv
from open3d import *
from PIL import Image

import utils

GPU = imp.load_source('GPUManager', './GPUManager.py')
TSDFtk = imp.load_source('TSDFtk', './TSDFManager.py')
My_MT = imp.load_source('MarchingTetrahedra', './MarchingTetrahedra.py')
Warp = imp.load_source('warpVolume', './warpVolume.py')
Body = imp.load_source('HumanBody', './Body.py')



def saveply(path, V, F=None):

    with open(path, "w") as f:
        f = open(path, "w")
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {0:d}\n".format(len(V)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("element face %d \n" %(len(F)))
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        # Vertices
        for vert in V:
            f.write("{0:f} {1:f} {2:f}\n".format(vert[0], vert[1], vert[2]))
        # Faces
        if not F is None:
            for face in F:
                f.write("3 {} {} {}\n".format(face[0], face[1], face[2]))


def load_pose_txt(parampath):
    pose = []
    with open(parampath) as f:
        lines = f.readlines()
        for line in lines:
            angles = line.split()
            pose += [[float(angles[0]), float(angles[1]), float(angles[2])]]
    pose = np.array(pose, dtype=np.float32)
    return pose


def colorize_mesh(Vertices, cimg):
    rows = cimg.shape[0]
    cols = cimg.shape[1]
    color_intrinsic = [525.898, 524.277, 490.843, 272.345]
    RT_c2d = np.array([[0.999976, -0.00643662, -0.0027378, 41.057],
                       [0.00644326, 0.999976, 0.00242119, 1.23825],
                       [0.00272215, -0.00243877, 0.999993, 1.41714],
                       [0, 0, 0, 1]], dtype=np.float32)

    # Calculate vertex normals
    indices = np.arange(len(Vertices), dtype=np.int32).reshape(-1,3)
    vertFaces = Vertices[indices]
    vecAB = vertFaces[:,1] - vertFaces[:,0]
    vecAC = vertFaces[:,2] - vertFaces[:,0]

    faceNormals = np.cross(vecAB, vecAC)
    vertNormals = np.zeros((len(Vertices),3))
    nomalCount = np.zeros(len(Vertices))
    for vset, facenormal in zip(indices, faceNormals):
        for j in vset:
            vertNormals[j] = (vertNormals[j] * nomalCount[j] + facenormal)/(nomalCount[j] + 1)
            nomalCount[j] += 1
    norms = LA.norm(vertNormals, axis=1)
    vertNormals = vertNormals / np.array([norms,norms,norms]).T


    # Project Vertices to colorimg
    VertexArray_2dproj = ((Vertices[:,0:2]/Vertices[:,2].reshape(-1, 1))*np.array([color_intrinsic[0], color_intrinsic[1]]) + np.array([color_intrinsic[2], color_intrinsic[3]])).astype(np.int32)
    conditions = np.array([0<=VertexArray_2dproj[:,1], VertexArray_2dproj[:,1]<rows, 0<=VertexArray_2dproj[:,0], VertexArray_2dproj[:,0]<cols])
    xlist = np.where(conditions.all(axis=0), VertexArray_2dproj[:,0], 0.0)
    ylist = np.where(conditions.all(axis=0), VertexArray_2dproj[:,1], 0.0)
    cimg[0,0,:] = 0
    BGR = cimg[ylist.astype(np.int32), xlist.astype(np.int32)]

    return BGR


def tsdf2mesh(tsdfpath, parampath, savepath, smplscale=1.0, colorpath=None):

    # Load TSDF
    TSDF = utils.loadTSDF_bin(tsdfpath)


    # Generate deformed coarse model
    voxelpath = "../coarsehuman/models/TshapeCoarseTetraD.ply"
    v_mesh = pymesh.load_mesh(voxelpath)

    VoxCnt = v_mesh.num_voxels

    voxelArray = np.array(v_mesh.voxels)
    vertexArray = np.array(v_mesh.vertices, dtype=np.float32)
    vertexvoxelArray = vertexArray[voxelArray].reshape(voxelArray.shape[0],1,12)[:,0].astype(np.float32)

    weights = np.load('../coarsehuman/models/coarseweights.npy') #weights of the coarse tetrahedra volume interpolated by SMPL body
    J = np.load('../coarsehuman/models/TshapeCoarseJoints.npy')
    J_shapedir = np.load("../coarsehuman/models/J_shapedir.npy") # J_shapedir (can deform joints depend on 10 betas)
    
    if splitext(parampath)[1] == ".pkl":
        with open(parampath, mode="rb") as f:
            param = pickle.load(f)
        trans =  param['trans']
        pose = param['pose']
        betas = param['betas']

    elif splitext(parampath)[1] == ".txt":
        pose = load_pose_txt(parampath)
        trans = np.zeros(3, dtype=np.float32)
        betas = np.zeros(10, dtype=np.float32)
    else:
        pose = np.zeros([24,3], dtype=np.float32)
        trans = np.zeros(3, dtype=np.float32)
        betas = np.zeros(10, dtype=np.float32)

    # Deform coarse model
    kintree_table = np.array([[4294967295,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21],
    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]])

    vertexArrayD, jointArrayD = Warp.warpVolume(vertexArray, J, J_shapedir, pose, betas, kintree_table, weights)

    vertexArrayD = (vertexArrayD + trans) * smplscale


    #Size of the TSDF map
    Size = np.array([1, 4, VoxCnt]) 
    GPUManager = GPU.GPUManager()
    # start = time.time()
    # V, F = mtetra.marchingTetrahedra(TSDF, voxelArray, vertexArrayD)
    # print("Time: {}".format(time.time() - start))
    # saveply(savepath, V, F)

    start = time.time()
    MarchingTetrahedra = My_MT.MarchingTetrahedra(Size, TSDF, 0.0, vertexvoxelArray, voxelArray, vertexArrayD, GPUManager)
    MarchingTetrahedra.run_CPU(TSDF, 0.0, voxelArray, vertexArrayD)

    colorize = True
    if colorize == True and colorpath is not None:
        if not os.path.exists(colorpath):
            MarchingTetrahedra.SaveToPly_CPU(savepath)
        else:
            cimg = cv2.imread(colorpath, -1)
            BGR = colorize_mesh(MarchingTetrahedra.Vertices_CPU, cimg)
            MarchingTetrahedra.SaveToPly_CPU(savepath, BGR=BGR)
    else:
        MarchingTetrahedra.SaveToPly_CPU(savepath)
    print("Time: {}".format(time.time() - start))
    
    saveply(splitext(savepath)[0] + "_coarse.ply", vertexArrayD, v_mesh.faces)
    




if __name__ == "__main__":

    parser = ArgumentParser(description='Create mesh from TSDF')
    parser.add_argument(
        '--tsdfpath',
        type=str,
        default=None,
        help='Path to TSDF')
    parser.add_argument(
        '--parampath',
        type=str,
        default=None,
        help='Path to SMPL param')
    parser.add_argument(
        '--savedir',
        type=str,
        default="",
        help='Dirpath to save result mesh')
    args = parser.parse_args()

    tsdfpath = args.tsdfpath

    if tsdfpath is None:
        print("Select TSDFs")

        root = tk.Tk()
        root.withdraw()
        tsdfpath = tkFileDialog.askopenfilenames()
        root.destroy()
        if tsdfpath == ():
            exit()

    for path in tsdfpath:
        print("TSDF path: " + path)
    
    parampath = args.parampath

    if parampath is None:
        print("Select SMPL param")

        root = tk.Tk()
        root.withdraw()
        parampath = tkFileDialog.askopenfilenames()
        root.destroy()
        if parampath == ():
            exit()
    for path in parampath:
        print("SMPLparam path: " + path)
            
        
    
    if not len(tsdfpath) == len(parampath):
        parampath = ["dummy" for i in range(len(tsdfpath))]
    for i in range(len(tsdfpath)):
            
        savedir = args.savedir
        if savedir == "":
            savedir = dirname(tsdfpath[i])
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        print("Save dir: " + savedir)
        savepath = savedir + "/{}.ply".format(splitext(basename(tsdfpath[i]))[0])
        tsdf2mesh(tsdfpath[i], parampath[i], savepath) 

