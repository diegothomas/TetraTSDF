import os
import numpy as np
import pymesh 
from argparse import ArgumentParser
import Tkinter as tk
import tkFileDialog
from tqdm import tqdm

import chumpy as ch
import smpl_webuser
from smpl_webuser.serialization import load_model
from warpVolume import unwarpVolume



def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)

def findSMPLcorrespondence(tetravertices,smplv):
    num_tetravertices = len(tetravertices)
    resultCoords = np.zeros((num_tetravertices,3))
    resultIds = np.zeros(num_tetravertices,dtype=np.int16)  
    # for i,v in tqdm(enumerate(coarse.vertices)):
    for i in tqdm(range(num_tetravertices)):
        v = tetravertices[i]
        vertid = closest_node(v,smplv)
        resultCoords[i] = smplv[vertid]
        resultIds[i] = vertid
    return resultCoords,resultIds

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

def createweights(tetrapath, savepath, gender, tetrapose):

    tetramesh = pymesh.load_mesh(tetrapath)
    num_vertices = len(tetramesh.vertices)

    # Select gender
    if gender=="male":
        print "Use male smpl model"
        smpl = load_model( './models/basicModel_m_lbs_10_207_0_v1.0.0.pkl' )
        PATH_J_onbetas = "./models/J_onbetas_male/"
    elif gender=="female":
        print "Use female smpl model"
        smpl = load_model( './models/basicModel_f_lbs_10_207_0_v1.0.0.pkl' )
        PATH_J_onbetas = "./models/J_onbetas_female/"
    else:
        print "Use neutral smpl model"
        smpl = load_model( './models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl' )
        PATH_J_onbetas = "./models/J_onbetas_neutral/"

    sv = smpl_webuser.verts.verts_decorated(
        trans = ch.zeros(3),
        pose=ch.array(tetrapose),
        #pose=ch.zeros(24*3),
        v_template=smpl.v_template,
        J=smpl.J_regressor,
        # betas=model.betas,
        betas=ch.zeros(10),
        shapedirs=smpl.shapedirs[:, :, :10],
        weights=smpl.weights,
        kintree_table=smpl.kintree_table,
        bs_style=smpl.bs_style,
        f=smpl.f,
        bs_type=smpl.bs_type,
        posedirs=smpl.posedirs)
    

    print "num of tetra vertices: ", num_vertices

    print "findSMPLcorrespondence..." 
    # corrCoords = np.zeros((num_vertices,3))     
    # corrIds = np.zeros((num_vertices),dtype=np.int16)
    corrCoords,corrIds = findSMPLcorrespondence(tetramesh.vertices, sv.r)
    print "Done."
    
    print "findSMPLweights...", 
    # coarseweights = findSMPLweights(num_vertices,corrIds,m.weights)
    coarseweights = findSMPLweights(num_vertices,corrIds,smpl.weights)
    print "Done."

    print "findSMPLposedirs...", 
    coarseposedirs = np.zeros((num_vertices,3, 207),dtype=np.float32)
    coarseposedirs = findSMPLposedirs(num_vertices,corrIds,smpl.posedirs)
    print "Done."

    # Create shapedir for smpl joint
    # J_shapedir = np.empty([24, 3, 10])
    # for i in range(10):
    #     j_onbetas = pymesh.load_mesh(PATH_J_onbetas + "J_onbetas_{}.ply".format(i))
    #     J_shapedir[:,:,i] = j_onbetas.vertices - smpl.J

    J_shapedir = np.dstack([smpl.J_regressor.dot(smpl.shapedirs[:, :, i]) for i in range(10)])

    savepath
    print "Saving weights to: {}...".format(savepath),
    np.save(savepath + '/coarseposedirs.npy', coarseposedirs)
    np.save(savepath + '/coarseweights.npy', coarseweights)
    # np.save(savepath + '/corrCoords.npy', corrCoords)
    # np.save(savepath + '/corrIds.npy', corrIds)
    np.save(savepath + '/TshapeCoarseJoints.npy', smpl.J)
    np.save(savepath + "/J_shapedir", J_shapedir)
    print "Done."

    # Deform Tetrahedral Coarse model to T pose by using created weights
    coarse_Tpose = unwarpVolume(tetramesh.vertices, sv.J, tetrapose, smpl.kintree_table, coarseweights)
    pymesh.meshio.save_mesh_raw(savepath + "/TshapeCoarseTetraD.ply", coarse_Tpose, tetramesh.faces, tetramesh.voxels)
    


if __name__ == '__main__':
    parser = ArgumentParser(description='Create weights')
    parser.add_argument(
        '--tetrapath',
        type=str,
        default='./CoarseTetra.ply',
        help='Path to tetrahedralized coarse human')
    parser.add_argument(
        '--savepath',
        type=str,
        default="./models/",
        help='Path to save created parameters, ')
    parser.add_argument(
        '--gender',
        type=str,
        default="male",
        help='Gender of the smpl model')
    parser.add_argument(
        '--starpose',
        action="store_true",
        help='Whether use star-posed coarse human. leg-opened model prevents mismatch of the closest nodes')

    args = parser.parse_args()

       
    tetrapath = args.tetrapath
    if not os.path.exists(tetrapath):
        print "Select tetrahedralized coarse human"

        root = tk.Tk()
        root.withdraw()
        tetrapath = tkFileDialog.askopenfilename()
        root.destroy()
        if tetrapath == ():
            exit()

    pose = np.zeros(24*3)
    if args.starpose:
        print("Option: --starpose True")
        print("Calculate weights on star-posed model")
        pose[3:6] = np.array([0,  0,  0.5])
        pose[6:9] = np.array([0,  0,  -0.5])
    else:
        print("Option: --starpose False")
        print("Calculate weights on T-posed model")
    createweights(tetrapath, args.savepath, args.gender, pose)