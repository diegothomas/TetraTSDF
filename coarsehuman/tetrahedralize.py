#PyMesh ile tetra yapma denemeleri yasasin ppymesh olleeyy olleeeyy
from argparse import ArgumentParser
import pymesh
import numpy as np
import Tkinter as tk
import tkFileDialog
import os
from os.path import join, exists, abspath, dirname, basename, splitext

import utils

from tqdm import tqdm


def make_adjlist(voxels, num_vertices):
    adjlist = [[] for i in range(num_vertices)]
    # for ids in voxels:
    for i in tqdm(range(len(voxels))):
        ids = voxels[i]
        ids_s = sorted(ids)
        adjlist[ids_s[0]] = list(set([ids_s[1], ids_s[2], ids_s[3]] + adjlist[ids_s[0]]))
        adjlist[ids_s[1]] = list(set([ids_s[0], ids_s[2], ids_s[3]] + adjlist[ids_s[1]]))
        adjlist[ids_s[2]] = list(set([ids_s[0], ids_s[1], ids_s[3]] + adjlist[ids_s[2]]))
        adjlist[ids_s[3]] = list(set([ids_s[0], ids_s[1], ids_s[3]] + adjlist[ids_s[3]]))
    
    return adjlist

def reduce_points(points, rate):
    Ids = [int(rate * i) for i in range(int(len(points)/rate))]
    return points[Ids]


def main(tetra_path, savedir, create_adjlists, adjnodes, rates):
    print "######### Tetrahedralize #########"
    mesh = pymesh.load_mesh(tetra_path)
    mesh.add_attribute("vertex_normal")
    mesh.get_attribute("vertex_normal")
    mesh.get_vertex_attribute("vertex_normal")

    #some meshes may have self intersection so tetgen cannot deal with it.
    #Before tetrahedralization here is the self intersection cleanup
    self_mesh = pymesh.resolve_self_intersection(mesh, engine='auto')
    self_mesh.add_attribute("vertex_normal")
    self_mesh.get_attribute("vertex_normal")
    self_mesh.get_vertex_attribute("vertex_normal")

    print("Starting Tetgen.............................")
    tetgen = pymesh.tetgen()
    tetgen.points = self_mesh.vertices
    tetgen.triangles = self_mesh.faces
    # tetgen.max_tet_volume = 0.000001 #Very detailed
    tetgen.max_tet_volume = 0.0000005 #Very detailed
    tetgen.verbosity = 0
    tetgen.run()

    outmesh = tetgen.mesh
    print(outmesh.num_vertices, outmesh.num_faces,outmesh.num_voxels)
    print(outmesh.dim, outmesh.vertex_per_face,outmesh.vertex_per_voxel)

    # Save tetrahedral mesh
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    pymesh.meshio.save_mesh(savedir + "/CoarseTetra.ply", outmesh)



    # Option: Create adjlists for tetranet
    if create_adjlists:
        print "#########Create adjlists#########"
        
        # Reduce vertices for full connection

        coarse1 = reduce_points(outmesh.vertices, rates[0])
        utils.save_points(coarse1, savedir + "/coarse1.ply")
        coarse2 = reduce_points(coarse1, rates[1])
        utils.save_points(coarse2, savedir + "/coarse2.ply")
        coarse3 = reduce_points(coarse2, rates[2])
        utils.save_points(coarse3, savedir + "/coarse3.ply")
        coarse4 = reduce_points(coarse3, rates[3])
        utils.save_points(coarse4, savedir + "/coarse4.ply")


        print("Generate node adj list")
        # Note: utils.make_adjlist returns a list of n nearest nodes (L2 norm)
        print "Node reduction rates: ",
        print(rates)
        n = adjnodes
        print("Adjacent Nodes: " + str(n))

        # For partial connection
        print("coarse 4 to 3: {} -> {}".format(len(coarse4), len(coarse3)))
        adjlist = utils.make_adjlist(coarse4, coarse3, n, True)
        utils.list2csv(adjlist, savedir + "/adjlist_4to3.csv")
        print("coarse 3 to 2: {} -> {}".format(len(coarse3), len(coarse2)))
        adjlist = utils.make_adjlist(coarse3, coarse2, n, True)
        utils.list2csv(adjlist, savedir + "/adjlist_3to2.csv")
        print("coarse 2 to 1: {} -> {}".format(len(coarse2), len(coarse1)))
        adjlist = utils.make_adjlist(coarse2, coarse1, n, True)
        utils.list2csv(adjlist, savedir + "/adjlist_2to1.csv")
        print("coarse 1 to original: {} -> {}".format(len(coarse1), len(outmesh.vertices)))
        adjlist = utils.make_adjlist(coarse1, outmesh.vertices, n, True)
        utils.list2csv(adjlist, savedir + "/adjlist_1to0.csv")

        # # For GCN 
        # print("self connection 4: {} -> {}".format(len(coarse4), len(coarse4)))
        # adjlist = utils.make_adjlist(coarse4, coarse4, n,)
        # utils.list2csv(adjlist, savedir + "/adjlist_4.csv")
        # print("self connection 3: {} -> {}".format(len(coarse3), len(coarse3)))
        # adjlist = utils.make_adjlist(coarse3, coarse3, n)
        # utils.list2csv(adjlist, savedir + "/adjlist_3.csv")
        # print("self connection 2: {} -> {}".format(len(coarse2), len(coarse2)))
        # adjlist = utils.make_adjlist(coarse2, coarse2, n)
        # utils.list2csv(adjlist, savedir + "/adjlist_2.csv")
        # print("self connection 1: {} -> {}".format(len(coarse1), len(coarse1)))
        # adjlist = utils.make_adjlist(coarse1, coarse1, n)
        # utils.list2csv(adjlist, savedir + "/adjlist_1.csv")
        # print("self connection original: {} -> {}".format(len(outmesh.vertices), len(outmesh.vertices)))
        # adjlist = make_adjlist(outmesh.voxels, outmesh.num_vertices)
        # utils.list2csv(adjlist, savedir + "/adjlist_0.csv")





if __name__ == '__main__':
    parser = ArgumentParser(description='Tetrahedralize inside of the mesh and create node adjacency lists')
    parser.add_argument(
        '--plypath',
        type=str,
        default="",
        # default="./coarsehuman_v2_3_blender.ply",
        help='Path to mesh .ply')
    parser.add_argument(
        '--savedir',
        type=str,
        default="./",
        help='Path to save results')
    parser.add_argument(
        '--create_adjlists',
        action="store_true",
        help='Add option: Create adjlists for tetranet after tetrahedralization')
    parser.add_argument(
        '--adjnodes',
        type=int,
        default=9,
        help='Add option: number of adjacent nodes')
    parser.add_argument(
        '--rates',
        nargs="+",
        type=float,
        default=[4.0,2.0,2.0,2.0],
        help='Add option: reduction rates of adjacent nodes')

    args = parser.parse_args()

       
    plypath = args.plypath

    if plypath == "":
        print "Select ply to tetrahedralize"

        root = tk.Tk()
        root.withdraw()
        plypath = tkFileDialog.askopenfilename()
        root.destroy()
        if plypath == ():
            exit()
            
    # rates = [4,2,2,2]   #Original
    # rates = [4,1.5,1.5,1.5]   #thick network
    # rates = [6,3,2,2]   # upper thin network
    # rates = [8,2,2,2]   #upper thin network2
    # rates = [4,2,3,4]   #lower thin network

    main(plypath, args.savedir, args.create_adjlists, args.adjnodes, args.rates)

