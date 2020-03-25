import pymesh 
import numpy as np
from argparse import ArgumentParser
import Tkinter as tk
import tkFileDialog
from tqdm import tqdm

import chumpy as ch
import smpl_webuser
from smpl_webuser.serialization import load_model
from warpVolume import unwarpVolume
from os.path import dirname
import cPickle as pickle


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)

def findSMPLcorrespondence(tetravertices,smplv):
    num_tetravertices = len(tetravertices)
    result = np.zeros((num_tetravertices,3))
    resultloc = np.zeros(num_tetravertices,dtype=np.int16)  
    # for i,v in tqdm(enumerate(coarse.vertices)):
    for i in tqdm(range(num_tetravertices)):
        v = tetravertices[i]
        vertid = closest_node(v,smplv)
        result[i] = smplv[vertid]
        resultloc[i] = vertid
    return result,resultloc

def findSMPLweights(coarsevertcount,corr,smplweights):
    # newweights = np.zeros((coarsevertcount, 24),dtype=np.float32)
    # for i,idv in enumerate(corr):
    #     newweights[i] = smplweights[idv]
    newweights = smplweights[corr]
    newweights = np.array(newweights, dtype=np.float32)
    return newweights

def findSMPLposedirs(coarsevertcount,corr,smplposedir):
    # newposedirs = np.zeros((coarsevertcount,3, 207),dtype=np.float32)
    # for i,idv in enumerate(corr):
        # newposedirs[i,:,:] = smplposedir[idv,:,:] 
        # newposedirs = np.vstack((newposedirs, poselist[idv])) 
    newposedirs = smplposedir[corr,:,:]
    newposedirs = np.array(newposedirs, dtype=np.float32)
    return newposedirs

def createweights(meshpath, targetsmplpath, parampath, savedir, gender):

    targetmesh = pymesh.load_mesh(meshpath)
    targetsmpl = pymesh.load_mesh(targetsmplpath)
    num_vertices = len(targetmesh.vertices)

    with open(parampath, "rb") as f:
        param = pickle.load(f)
    pose = param["pose"]
    trans = param["trans"]
    betas = param["betas"]



    # Select gender
    if gender=="male":
        print "Use male smpl model"
        smpl = load_model( './basicModel_m_lbs_10_207_0_v1.0.0.pkl' )
    elif gender=="female":
        print "Use female smpl model"
        smpl = load_model( './basicModel_f_lbs_10_207_0_v1.0.0.pkl' )
    else:
        print "Use neutral smpl model"
        smpl = load_model( './basicModel_neutral_lbs_10_207_0_v1.0.0.pkl' )

    sv = smpl_webuser.verts.verts_decorated(
        trans = ch.array(trans),
        pose=ch.array(pose),
        #pose=ch.zeros(24*3),
        v_template=smpl.v_template,
        J=smpl.J_regressor,
        betas=ch.array(betas),
        # betas=ch.zeros(10),
        shapedirs=smpl.shapedirs[:, :, :10],
        weights=smpl.weights,
        kintree_table=smpl.kintree_table,
        bs_style=smpl.bs_style,
        f=smpl.f,
        bs_type=smpl.bs_type,
        posedirs=smpl.posedirs)
    
    pymesh.meshio.save_mesh_raw(savedir + "/test_smpl.ply", sv.r, smpl.f)

    print "num of target vertices: ", num_vertices

    print "findSMPLcorrespondence..." 
    corr,corrloc = findSMPLcorrespondence(targetmesh.vertices, targetsmpl.vertices)
    # corr,corrloc = findSMPLcorrespondence(targetmesh.vertices, sv.r)
    print "Done."
    
    print "findSMPLweights...",
    # targetweights = findSMPLweights(num_vertices,corrloc,m.weights)
    targetweights = findSMPLweights(num_vertices,corrloc,smpl.weights)
    print "Done."

    savepath = savedir + '/weights_smpl.npy'
    print "Saving weights to: " + savepath
    np.save(savepath, targetweights)
    print "Done."

    # Deform Tetrahedral Coarse model to T pose by using created weights
    target_Tpose = unwarpVolume(targetmesh.vertices, sv.J, pose, smpl.kintree_table, targetweights)
    pymesh.meshio.save_mesh_raw(savedir + "/test_deformed.ply", target_Tpose, targetmesh.faces, targetmesh.voxels)
    


if __name__ == '__main__':
    parser = ArgumentParser(description='Create weights')
    parser.add_argument(
        '--meshpath',
        type=str,
        default='',
        help='Path to input mesh')
    parser.add_argument(
        '--targetsmplpath',
        type=str,
        default='',
        help='Path to input smpl mesh')
    parser.add_argument(
        '--parampath',
        type=str,
        default="",
        help='Path to smpl parameter')
    parser.add_argument(
        '--savedir',
        type=str,
        default="",
        help='Path to save created weights')
    parser.add_argument(
        '--gender',
        type=str,
        default="male",
        help='Gender of the smpl model')
    args = parser.parse_args()

       
    meshpath = args.meshpath
    if meshpath == "":
        print "Select input mesh"

        root = tk.Tk()
        root.withdraw()
        meshpath = tkFileDialog.askopenfilename()
        root.destroy()
        if meshpath == ():
            exit()

    targetsmplpath = args.targetsmplpath
    if targetsmplpath == "":
        print "Select input smpl mesh"

        root = tk.Tk()
        root.withdraw()
        targetsmplpath = tkFileDialog.askopenfilename()
        root.destroy()
        if targetsmplpath == ():
            exit()

    parampath = args.parampath
    if parampath == "":
        print "Select smpl parameter"

        root = tk.Tk()
        root.withdraw()
        parampath = tkFileDialog.askopenfilename()
        root.destroy()
        if parampath == ():
            exit()
    savedir = args.savedir
    if savedir == "":
        savedir = dirname(meshpath)

    createweights(meshpath, targetsmplpath, parampath, savedir, args.gender)