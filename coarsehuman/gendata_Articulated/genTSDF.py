import numpy as np
import numpy.linalg as LA
import cv2
import os
import glob
import time
import struct
from argparse import ArgumentParser
from os.path import join, exists, abspath, dirname, basename, splitext
import skimage

import pymesh 
import cPickle as pickle
from tqdm import tqdm
import warpVolume

from pyflann import FLANN, set_distance_type

import Tkinter as tk
import tkFileDialog


def global_rigid_transformation(pose, kintree_table, J):

    results = {}

    pose = pose.reshape((-1,3))
    
    id_to_col = {kintree_table[1,i] : i for i in range(kintree_table.shape[1])}

    parent = {i : id_to_col[kintree_table[0,i]] for i in range(1, kintree_table.shape[1])}

    
    rodrigues = lambda x : cv2.Rodrigues(x)[0]
    rodriguesJB = lambda x : cv2.Rodrigues(x)[1]
    with_zeros = lambda x : np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))
    pack = lambda x : np.hstack([np.zeros((4, 3)), x.reshape((4,1))])
    

    results[0] = with_zeros(np.hstack((rodrigues(pose[0,:]), J[0,:].reshape((3,1)))))        

    for i in range(1, kintree_table.shape[1]):
        results[i] = results[parent[i]].dot(with_zeros(np.hstack((rodrigues(pose[i,:]),(J[i] - J[parent[i]]).reshape(3,1)))))

    Rt_Global = np.dstack([results[i] for i in sorted(results.keys())])
    Rt_A = np.dstack([results[i] - (pack(results[i].dot(np.concatenate( ( (J[i,:]), (0,) ) )))) for i in range(len(results))])
    
    return Rt_A, Rt_Global


def calcTSDF_point2plane(smplVerts, smplFaces, DCMVerts):

    # TODO
    # Improve TSDF calculation. (Current code generates noisy TSDF)

    # start = time.time()
    
    vertFaces = smplVerts[smplFaces]
    vecAB = vertFaces[:,1] - vertFaces[:,0]
    vecAC = vertFaces[:,2] - vertFaces[:,0]

    # Calculate vertex normals
    faceNormals = np.cross(vecAB, vecAC)
    vertNormals = np.zeros((len(smplVerts),3))
    nomalCount = np.zeros(len(smplVerts))
    for vset, facenormal in zip(smplFaces, faceNormals):
        for j in vset:
            vertNormals[j] = (vertNormals[j] * nomalCount[j] + facenormal)/(nomalCount[j] + 1)
            nomalCount[j] += 1
    norms = LA.norm(vertNormals, axis=1)
    vertNormals = vertNormals / np.array([norms,norms,norms]).T


    convVal = 32767.0
    nu = 0.03

    # Find nearest neighbor
    vertIds_list = []
    Dist_p2p_list = []
    TruncatedDist_list = []
    Mask_list = []
    for i in range(1):
        flann = FLANN()
        flann.build_index(smplVerts)
        vertIds, Dist_p2p = flann.nn_index(DCMVerts, num_neighbors=1)
        vertIds_list += [vertIds]
        Dist_p2p_list += [Dist_p2p]
    
    
        print(vertIds[0:10])

        Dist_p2p /= nu
        TruncatedDist_p2p = np.minimum(1.0, np.maximum(-1.0, Dist_p2p))


        # calculate TSDF
        D = vertNormals[:,0]*smplVerts[:,0] + vertNormals[:,1]*smplVerts[:,1] + vertNormals[:,2]*smplVerts[:,2]
        D = -D/LA.norm(vertNormals, axis=1)
        corrNormals = vertNormals[vertIds]
        Dist = corrNormals[:,0]*DCMVerts[:,0] + corrNormals[:,1]*DCMVerts[:,1] + corrNormals[:,2]*DCMVerts[:,2] + D[vertIds]
        Dist /= nu
        TruncatedDist = np.minimum(1.0, np.maximum(-1.0, Dist))

        # Mask = np.where(np.abs(TruncatedDist_p2p)>1.0, TruncatedDist_p2p/np.abs(TruncatedDist_p2p), TruncatedDist_p2p)
        # Mask = np.where(TruncatedDist_p2p>=1.0, 0, 1)
        # Mask = np.where((Dist_p2p / Dist)>1.5, 0, 1) * Mask
        # Mask = (np.where((Dist_p2p / Dist)>1.5, 0, 1) + Mask) - np.where((Dist_p2p / Dist)>1.5, 0, 1) * Mask
        # Mask = np.where((Dist / Dist_p2p)>1.5, 0, 1)
        Mask = TruncatedDist / np.abs(TruncatedDist)
        Mask_list += [Mask]
        
        # TruncatedDist = Mask*TruncatedDist - (1 - Mask)

        TruncatedDist_list += [TruncatedDist]
    # TruncatedDist = np.median(TruncatedDist_list, axis=0)
    # Mask = Mask_list[0]
    # for i in range(1, len(Mask_list)):
    #     Mask = Mask * Mask_list[i]
    # Mask = np.where(np.abs(np.sum(Mask_list, axis=0))<3, 0, 1)


    # TruncatedDist = Mask*TruncatedDist - (1 - Mask)
    # TruncatedDist = np.average(TruncatedDist_list, axis=0)

    # print("Time: {}".format(time.time() - start))
    TruncatedDist = TruncatedDist_list[0]
    return TruncatedDist * convVal
    # return np.array([TruncatedDist] + TruncatedDist_list) * convVal


def save_ply(fname, v, f=None, col=None, lab=None, fcol=None):
    cname = ('gray', 'red', 'blue', 'yellow', 'magenta', 'green', 'indigo', 'darkorange', 'cyan', 'pink', 'yellowgreen')
    cd = skimage.color.color_dict
    cols1 = [np.array(cd[cn]) * 255 for cn in cname]
    cols2 = [np.array(cd[cn]) * 127 for cn in cname]
    cols = cols1 + cols2

    if lab is not None:
        lset = set(list(lab[:]))
        ldic = {l: i for i, l in enumerate(lset)}

    n = v.shape[1]  # == c.shape[1]

    if f is not None:
        k = f.shape[1]
    else:
        k = 0
    with open(fname, 'w') as fout:
        fout.write('ply\n')
        fout.write('format ascii 1.0\n')
        fout.write('comment author: Greg Turk\n')
        fout.write('comment object: another cube\n')
        fout.write('element vertex %d\n' % n)
        fout.write('property float x\n')
        fout.write('property float y\n')
        fout.write('property float z\n')
        fout.write('property uchar red\n')
        fout.write('property uchar green\n')
        fout.write('property uchar blue\n')
        fout.write('element face %d\n' % k)
        fout.write('property list uchar int vertex_index\n')
        if np.any(fcol):
            fout.write('property uchar red\n')
            fout.write('property uchar green\n')
            fout.write('property uchar blue\n')
        fout.write('end_header\n')

        for i in range(n):
            if col is not None:
                c = col[:,i]
            else:
                if lab is not None:
                    c = cols[ldic[lab[i]] % len(cols)]
                else:
                    c = np.array((150, 150, 150))

            fout.write('%f %f %f %.0f %.0f %.0f\n' % (v[0, i], v[1, i], v[2, i], c[0], c[1], c[2]))

        if np.any(fcol):
            for i in range(k):
                fout.write('3 %d %d %d %d %d %d\n' % (f[0, i], f[1, i], f[2, i], fcol[0, i], fcol[1, i], fcol[2, i]))
        else:
            for i in range(k):
                fout.write('3 %d %d %d\n' % (f[0, i], f[1, i], f[2, i]))

    return


def saveTSDF_bin(TSDF, savepath):

	if not os.path.exists(os.path.dirname(savepath)):
		os.makedirs(os.path.dirname(savepath))

	with open(savepath, "wb") as f:
		num_nodes = TSDF.size
		f.write(struct.pack("I", num_nodes))
		f.write(struct.pack('<{:d}f'.format(TSDF.size), *TSDF))


def normalization_routine(artirootdir, subdir):

    print("########### normalization_routine GT mesh ###########")

    meshpaths = sorted(glob.glob(artirootdir + subdir + "/meshes/*"))
    fitpaths = sorted(glob.glob("./smplfitresults_articulated" + subdir + "/data/*.ply"))
    parampaths = sorted(glob.glob("./smplfitresults_articulated" + subdir + "/smplparams/*.pkl"))
    jointpaths = sorted(glob.glob("./smplfitresults_articulated" + subdir + "/joints/*.ply"))
    meshpaths.sort(key=lambda x:len(x))
    fitpaths.sort(key=lambda x:len(x))
    parampaths.sort(key=lambda x:len(x))
    jointpaths.sort(key=lambda x:len(x))

    print(len(meshpaths))
    print(len(fitpaths))
    print(len(parampaths))
    print(len(jointpaths))

    if not len(meshpaths)==len(fitpaths)==len(parampaths)==len(jointpaths):
        print("Invalid inputs")
        return


    savedir_gtmesh = artirootdir + subdir + "/meshes_centered"
    if not exists(savedir_gtmesh):
        os.makedirs(savedir_gtmesh)
    savedir_smplparam = artirootdir + subdir + "/smplparams_centered"
    if not exists(savedir_smplparam):
        os.makedirs(savedir_smplparam)

    for count, (meshpath, fitpath, parampath, jointpath) in enumerate(zip(meshpaths, fitpaths, parampaths, jointpaths)):
        if count % 10 == 0:
            print(meshpath)
            print(fitpath)
            print(parampath)
            print(jointpath)

        with open(parampath, mode="rb") as f:
            param = pickle.load(f)
        pose = param["pose"]
        pose = pose.ravel()

        
        # Turn the GTmesh to front
        GTmesh = pymesh.load_mesh(meshpath)
        T_front = cv2.Rodrigues(pose[0:3])[0].T


        # Calculate translation
        joints = pymesh.load_mesh(jointpath).vertices
        trans = joints[0] - np.dot(LA.inv(T_front), np.array([-0.00210668, -0.24589817,  0.02989279])) #root joint of male smplmodel 

        
        GT_normalized = np.dot(T_front, (GTmesh.vertices - trans).T)


        # Save nornalized GT mesh to .ply
        save_ply(savedir_gtmesh + '/mesh_{}_{}.ply'.format(basename(subdir), count), GT_normalized, f=GTmesh.faces.T)


        # Save smplparams to .pkl
        pose[0:3] = np.zeros(3)
        smplparams = {"pose": pose, "trans": np.zeros(3, dtype=np.float32), "betas": np.zeros(10, dtype=np.float32)}
        with open(savedir_smplparam + "/param_{}_{}.pkl".format(basename(subdir), count), "wb") as fb:
            pickle.dump(smplparams, fb)




def TSDFcalculation_routine(artirootdir, subdir, coarsemodeldir):

    print("########### Calculate TSDF ###########")


    coarseTetra_T = pymesh.load_mesh(coarsemodeldir + "/TshapeCoarseTetraD.ply") # Tetrahedralized coarse human (T-pose)
    coarseweights = np.load(coarsemodeldir + '/coarseweights.npy') # Joint weights for deformation of the tetrahedralized coarse human
    J = np.load(coarsemodeldir + '/TshapeCoarseJoints.npy') # Neutral joint coordinates
    J_shapedir = np.load(coarsemodeldir + "/J_shapedir.npy") # J_shapedir (can deform joints depend on 10 betas)
    kintree_table = np.array([[4294967295,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21],
                                [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]) # Define joint relationships
    
    weightpath = "./smplfitresults_articulated" + subdir + "/weights_smpl.npy"
    gtweights = np.load(weightpath)


    meshpaths = sorted(glob.glob(artirootdir + subdir + "/meshes_centered/*.ply"))
    parampaths = sorted(glob.glob(artirootdir + subdir + "/smplparams_centered/*.pkl"))
    meshpaths.sort(key=lambda x:len(x))
    parampaths.sort(key=lambda x:len(x))
        

    print(len(parampaths))
    print(len(meshpaths))
    if not len(meshpaths)==len(parampaths):
        print("Invalid inputs")
        return



    savedir_TSDF = artirootdir + subdir + "/TSDF"
    if not exists(savedir_TSDF):
        os.makedirs(savedir_TSDF)

    for count, (parampath, meshpath) in enumerate(zip(parampaths, meshpaths)):

        print(parampath)
        print(meshpath)
        start = time.time()
    

        # Load pose
        with open(parampath, mode="rb") as f:
            param = pickle.load(f)
        pose = param["pose"]
        pose = pose.ravel()

        # Define star-pose
        starpose = np.zeros(24*3)
        starpose[3:6] = np.array([0,  0,  0.5])
        starpose[6:9] = np.array([0,  0,  -0.5])

        # Generate rotation matrix
        A1, A1_global = global_rigid_transformation(pose, kintree_table, J)
        A1_inv = np.dstack([np.linalg.inv(A1[:,:,i]) for i in range(24)])
        A2, A2_global = global_rigid_transformation(starpose, kintree_table, J)

        R1 = A1_inv.dot(gtweights.T)
        R2 = A2.dot(gtweights.T)

        # Load GT mesh
        gtmesh = pymesh.load_mesh(meshpath)
        gtmesh_vertices = gtmesh.vertices

        # Deform GT mesh to star-pose
        gtmesh_vertices_4dim = np.vstack((gtmesh_vertices.T, np.ones((1, gtmesh_vertices.shape[0]))))
        v_Tpose = (R1[:,0] * gtmesh_vertices_4dim[0] + R1[:,1] * gtmesh_vertices_4dim[1] + R1[:,2] * gtmesh_vertices_4dim[2] + R1[:,3] * gtmesh_vertices_4dim[3])
        v_starpose = (R2[:,0] * v_Tpose[0] + R2[:,1] * v_Tpose[1] + R2[:,2] * v_Tpose[2] + R2[:,3] * v_Tpose[3]).T[:,:3]

        # Calculate TSDF
        # vertexArrayD, jointArrayD = warpVolume.warpVolume(coarseTetra_T.vertices, J, J_shapedir, pose, param["betas"], kintree_table, coarseweights)
        # TSDF = calcTSDF_point2plane(gtmesh.vertices, gtmesh.faces, vertexArrayD)
        vertexArrayD, jointArrayD = warpVolume.warpVolume(coarseTetra_T.vertices, J, J_shapedir, starpose, param["betas"], kintree_table, coarseweights)
        TSDF = calcTSDF_point2plane(v_starpose, gtmesh.faces, vertexArrayD)

        # Save TSDF
        savepath = savedir_TSDF + "/{}.bin".format(splitext(basename(meshpath))[0])
        saveTSDF_bin(TSDF, savepath)
        
        # save_ply(splitext(savepath)[0] + ".ply", v_starpose.T, f=gtmesh.faces.T, col=None, lab=None, fcol=None)


        print("Time: {}".format(time.time() - start))






if __name__ == "__main__":

    parser = ArgumentParser(description='Normalize fitting results')
    parser.add_argument(
        '--artirootdir',
        type=str,
        default=None,
        help='Dirpath to articulated dataset')
    parser.add_argument(
        '--coarsemodeldir',
        type=str,
        default="../models",
        help='Dirpath to tetrahedralized coarse model')
    args = parser.parse_args()

    artirootdir = args.artirootdir
    if artirootdir is None:
        print("Select articulated dataset root directory")
        root = tk.Tk()
        root.withdraw()
        artirootdir = tkFileDialog.askdirectory()
        root.destroy()
        if artirootdir == ():
            exit()

    if not exists(artirootdir):
        print("Invalid path to articulated dataset.")
        exit()

    # subdir = "/D_bouncing"
    # subdir = "/D_march"
    # subdir = "/I_crane"
    # subdir = "/I_jumping"

    subdirs = ["/D_bouncing", "/D_march", "/I_crane", "/I_jumping"]
    for subdir in subdirs:
        print("Subject: {}".format(subdir))
        normalization_routine(artirootdir, subdir)
        TSDFcalculation_routine(artirootdir, subdir, args.coarsemodeldir)