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

######test by onizuka######
import sys
sys.path.append("./smpl_webuser/")
###########################
import csv
import open3d as o3d
from PIL import Image
import utils
import pyflann

GPU = imp.load_source('GPUManager', './GPUManager.py')
TSDFtk = imp.load_source('TSDFtk', './TSDFManager.py')
My_MT = imp.load_source('MarchingTetrahedra', './MarchingTetrahedra.py')
Warp = imp.load_source('warpVolume', './warpVolume.py')
Body = imp.load_source('HumanBody', './Body.py')



def tsdf2chamfer(coarsemodel, targetpath_reg, targetpath_gt, parampath_reg, parampath_gt, savepath_reg, savepath_gt, use_mesh_reg=False, use_mesh_gt=False):


    # Generate deformed coarse model
    v_mesh = coarsemodel["v_mesh"]

    VoxCnt = v_mesh.num_voxels
    voxelArray = np.array(v_mesh.voxels)
    vertexArray = np.array(v_mesh.vertices, dtype=np.float32)
    vertexvoxelArray = vertexArray[voxelArray].reshape(voxelArray.shape[0],1,12)[:,0].astype(np.float32)

    weights = coarsemodel["weights"] #weights of the coarse tetrahedra volume interpolated by SMPL body
    J = coarsemodel["J"]
    J_shapedir = coarsemodel["J_shapedir"] # J_shapedir (can deform joints depend on 10 betas)
    kintree_table = coarsemodel["kintree_table"]
    
    # Load data
    param_reg = utils.load_pkl(parampath_reg)
    param_gt = utils.load_pkl(parampath_gt)
    if use_mesh_reg:
        mesh_reg = o3d.io.read_triangle_mesh(targetpath_reg)
        Vertices_reg = np.asarray(mesh_reg.vertices)
    else:
        tsdf_reg = utils.loadTSDF_bin(targetpath_reg)
        # Deform coarse model
        vertexArrayD_reg, jointArrayD_reg = Warp.warpVolume(vertexArray, J, J_shapedir, param_reg["pose"], param_reg["betas"], kintree_table, weights)
        Size = np.array([1, 4, VoxCnt]) 
        GPUManager = GPU.GPUManager()

        MarchingTetrahedra = My_MT.MarchingTetrahedra(Size, tsdf_reg, 0.0, vertexvoxelArray, voxelArray, vertexArrayD_reg, GPUManager)
        Vertices_reg = MarchingTetrahedra.run_CPU(tsdf_reg, 0.0, voxelArray, vertexArrayD_reg)
        MarchingTetrahedra.SaveToPly_CPU(savepath_reg)

    if use_mesh_gt:
        mesh_gt = o3d.io.read_triangle_mesh(targetpath_gt)
        Vertices_gt = np.asarray(mesh_gt.vertices)
    else:
        tsdf_gt = utils.loadTSDF_bin(targetpath_gt)
        vertexArrayD_gt, jointArrayD_gt = Warp.warpVolume(vertexArray, J, J_shapedir, param_gt["pose"], param_gt["betas"], kintree_table, weights)
        Size = np.array([1, 4, VoxCnt]) 
        GPUManager = GPU.GPUManager()

        MarchingTetrahedra = My_MT.MarchingTetrahedra(Size, tsdf_gt, 0.0, vertexvoxelArray, voxelArray, vertexArrayD_gt, GPUManager)
        Vertices_gt = MarchingTetrahedra.run_CPU(tsdf_gt, 0.0, voxelArray, vertexArrayD_gt)
        MarchingTetrahedra.SaveToPly_CPU(savepath_gt)
        
    # flann = pyflann.FLANN()
    # pyflann.set_distance_type('euclidean')
    # flann.build_index(Vertices_gt, algorithm='kmeans', centers_init='kmeanspp', random_seed=1984)
    # vertIds, Dists = flann.nn_index(Vertices_reg, num_neighbors=1)
    # chamfer_pyflann = np.average(Dists)*100

    GTpc = o3d.geometry.PointCloud()
    GTpc.points = o3d.utility.Vector3dVector(Vertices_gt)
    regpc = o3d.geometry.PointCloud()
    regpc.points = o3d.utility.Vector3dVector(Vertices_reg)

    dists = np.array(regpc.compute_point_cloud_distance(GTpc))
    chamfer_o3d = np.average(dists)


    return chamfer_o3d
    # return chamfer_o3d, chamfer_pyflann


def tsdf2chamfer_fortex2shape(target_v, tsdf_gt, param_reg, param_gt, savepath1, savepath2):


    # Generate deformed coarse model
    voxelpath = "./models/TshapeCoarseTetraD.ply"
    v_mesh = pymesh.load_mesh(voxelpath)

    VoxCnt = v_mesh.num_voxels

    voxelArray = np.array(v_mesh.voxels)
    vertexArray = np.array(v_mesh.vertices, dtype=np.float32)
    vertexvoxelArray = vertexArray[voxelArray].reshape(voxelArray.shape[0],1,12)[:,0].astype(np.float32)

    weights = np.load('./models/coarseweights.npy') #weights of the coarse tetrahedra volume interpolated by SMPL body
    J = np.load('./models/Tshapecoarsejoints.npy')
    J_shapedir = np.load("./models/J_shapedir.npy") # J_shapedir (can deform joints depend on 10 betas)
    

    # Deform coarse model
    kintree_table = np.array([[4294967295,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21],
    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]])

    vertexArrayD_gt, jointArrayD_gt = Warp.warpVolume(vertexArray, J, J_shapedir, param_gt["pose"], param_gt["betas"], kintree_table, weights)


    Size = np.array([1, 4, VoxCnt]) 
    GPUManager = GPU.GPUManager()


    MarchingTetrahedra = My_MT.MarchingTetrahedra(Size, tsdf_gt, 0.0, vertexvoxelArray, voxelArray, vertexArrayD_gt, GPUManager)
    Vertices_gt = MarchingTetrahedra.run_CPU(tsdf_gt, 0.0, voxelArray, vertexArrayD_gt)
    MarchingTetrahedra.SaveToPly_CPU(savepath2)

    pyflann.set_distance_type("euclidean")
    flann = FLANN()
    flann.build_index(Vertices_gt, algorithm='kmeans', centers_init='kmeanspp', random_seed=1984)
    vertIds, dists = flann.nn_index(target_v, num_neighbors=1)
    chamfer_pyflann = np.average(dists)

    sourcepc = o3d.geometry.PointCloud()
    sourcepc.points = o3d.utility.Vector3dVector(Vertices_gt)
    targetpc = o3d.geometry.PointCloud()
    targetpc.points = o3d.utility.Vector3dVector(target_v)

    dists = np.array(targetpc.compute_point_cloud_distance(sourcepc))
    chamfer_o3d = np.average(dists)


    return chamfer_o3d, chamfer_pyflann


    


if __name__ == "__main__":


    # Use mesh as target
    use_mesh = True


    parser = ArgumentParser(description='Calculate average chamfer distance from TSDF')
    parser.add_argument(
        '--TSDFdir_pred',
        type=str,
        default=None,
        help='Path to regressed TSDF')
    parser.add_argument(
        '--TSDFdir_GT',
        type=str,
        default=None,
        help='Path to ground truth TSDF')
    parser.add_argument(
        '--paramdir_pred',
        type=str,
        default=None,
        help='Path to regressed SMPL params')
    parser.add_argument(
        '--paramdir_GT',
        type=str,
        default=None,
        help='Path to ground truth SMPL params')
    args = parser.parse_args()

    TSDFdir_pred = args.TSDFdir_pred

    if TSDFdir_pred is None:
        print("Select predicted TSDF dir")

        root = tk.Tk()
        root.withdraw()
        TSDFdir_pred = tkFileDialog.askdirectory()
        root.destroy()
        if TSDFdir_pred == ():
            exit()
    print("Predicted TSDF dirpath: " + TSDFdir_pred)

    TSDFdir_GT = args.TSDFdir_GT

    if TSDFdir_GT is None:
        print("Select GT TSDF dir")

        root = tk.Tk()
        root.withdraw()
        TSDFdir_GT = tkFileDialog.askdirectory()
        root.destroy()
        if TSDFdir_GT == ():
            exit()
    print("GT TSDF dirpath: " + TSDFdir_GT)
    
    paramdir_pred = args.paramdir_pred

    if paramdir_pred is None:
        print("Select SMPL param dir for predicted TSDFs")

        root = tk.Tk()
        root.withdraw()
        paramdir_pred = tkFileDialog.askdirectory()
        root.destroy()
        if paramdir_pred == ():
            exit()
    print("Regressed SMPLparam dirpath: " + paramdir_pred)


    paramdir_GT = args.paramdir_GT

    if paramdir_GT is None:
        print("Select GT SMPL param dir")

        root = tk.Tk()
        root.withdraw()
        paramdir_GT = tkFileDialog.askdirectory()
        root.destroy()
        if paramdir_GT == ():
            exit()
    print("GT SMPLparam dirpath: " + paramdir_GT)
            
    savedir = dirname(TSDFdir_pred) + "/chamfer_result"
        
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    targetpath_Regs = sorted(glob.glob(TSDFdir_pred + "/*.bin"))
    targetpath_Regs.sort(key=lambda x:len(x))
    targetpath_GT = sorted(glob.glob(TSDFdir_GT + "/*.bin"))
    targetpath_GT.sort(key=lambda x:len(x))
    parampath_Regs = sorted(glob.glob(paramdir_pred + "/*.pkl"))
    parampath_Regs.sort(key=lambda x:len(x))
    parampath_GT = sorted(glob.glob(paramdir_GT + "/*.pkl"))
    parampath_GT.sort(key=lambda x:len(x))
    use_mesh_reg = False
    use_mesh_GT = False
    if len(targetpath_Regs) == 0:
        targetpath_Regs = sorted(glob.glob(TSDFdir_pred + "/*.ply") + glob.glob(TSDFdir_pred + "/*.obj"))
        targetpath_Regs.sort(key=lambda x:len(x))
        use_mesh_reg = True
    if len(targetpath_GT) == 0:
        targetpath_GT = sorted(glob.glob(TSDFdir_GT + "/*.ply") + glob.glob(TSDFdir_GT + "/*.obj"))
        targetpath_GT.sort(key=lambda x:len(x))
        use_mesh_GT = True



    len_list = [len(targetpath_Regs), len(targetpath_GT), len(parampath_Regs), len(parampath_GT)]
    if not len_list[0]==len_list[1]==len_list[2]==len_list[3]:
        print("Invalid number of data")
        sys.exit()



    # Load deformed coarse model
    voxelpath = "../coarsehuman/models/TshapeCoarseTetraD.ply"
    v_mesh = pymesh.load_mesh(voxelpath)
    weights = np.load('../coarsehuman/models/coarseweights.npy') #weights of the coarse tetrahedra volume interpolated by SMPL body
    J = np.load('../coarsehuman/models/TshapeCoarseJoints.npy')
    J_shapedir = np.load("../coarsehuman/models/J_shapedir.npy") # J_shapedir (can deform joints depend on 10 betas)
    kintree_table = np.array([[4294967295,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21],
    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]])
    
    coarsemodel = {"v_mesh": v_mesh, "weights":weights, "J":J, "J_shapedir":J_shapedir, "kintree_table":kintree_table}


    avg_chamfer_o3d = 0
    avg_chamfer_pyflann = 0

    count = 0
    for i in range(len_list[0]):
    # for i in range(10):
        targetpath_reg = targetpath_Regs[i]
        targetpath_gt = targetpath_GT[i]
        parampath_reg = parampath_Regs[i]
        parampath_gt = parampath_GT[i]
        savepath_reg = savedir + "/{}_reg.ply".format(i)
        savepath_gt = savedir + "/{}_gt.ply".format(i)


        # chamfer_o3d, chamfer_pyflann = tsdf2chamfer(coarsemodel, targetpath_reg, targetpath_gt, parampath_reg, parampath_gt, savepath_reg, savepath_gt, use_mesh_reg, use_mesh_GT)
        # avg_chamfer_pyflann = (avg_chamfer_pyflann*count + chamfer_pyflann) / (count + 1)
        # print("Chamfer {:>4d}: Open3D: {:>.4f}, average: {:>.4f}, Pyflann: {:>.4f}, average: {:>.4f}".format(i, chamfer_o3d, avg_chamfer_o3d, chamfer_pyflann, avg_chamfer_pyflann))

        chamfer_o3d = tsdf2chamfer(coarsemodel, targetpath_reg, targetpath_gt, parampath_reg, parampath_gt, savepath_reg, savepath_gt, use_mesh_reg, use_mesh_GT)
        if chamfer_o3d > 0.1:
            print("Invalid value")
            continue
        avg_chamfer_o3d = (avg_chamfer_o3d*count + chamfer_o3d) / (count + 1)
        print("Chamfer {:>4d}: Open3D: {:>.4f}, average: {:>.4f}".format(i, chamfer_o3d, avg_chamfer_o3d))
        count += 1
