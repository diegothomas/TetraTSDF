import csv
from tqdm import tqdm
import numpy as np
import struct
import os
from os.path import exists, basename, splitext
import binascii
import pymesh
import skimage.color
import cPickle as pickle


def list2csv(dlist, path):
    with open(path, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(dlist)

def save_points(V, path):
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

def closest_node(node, nodes, num_nearest_nodes=9):
    adj = []
    nodes_np = np.asarray(nodes).copy()
    nodes_np.flags.writeable = True

    for i in range(num_nearest_nodes):
        dist_2 = np.sum((nodes_np - node)**2, axis=1)
        nearest_idx = np.argmin(dist_2)
        adj += [nearest_idx]
        nodes_np[nearest_idx] = nodes_np[np.argmax(dist_2)]

    return adj


def make_adjlist(points_s, points_t, num_nearest_nodes=9, allow_self_connection=False):

    # find adjlist for each vertices in points_t
    adjlist = []
    
    for i in tqdm(range(len(points_t))):
        v = points_t[i]
        adj = closest_node(v, points_s, num_nearest_nodes)
        if not allow_self_connection:
            try:
                adj.remove(i)
            except ValueError:
                pass
        adjlist += [adj]
    return adjlist

def make_adjlist_random_connection(num_points_s, num_points_t, num_nearest_nodes=9, allow_self_connection=False):

    # find adjlist for each vertices in points_t
    adjlist = []
    import random
    for i in tqdm(range(num_points_t)):
        adj = random.sample(xrange(num_points_s), num_nearest_nodes)
        adjlist += [adj]
    return adjlist

def loadTSDF_bin(filepath):

	with open(filepath, "rb") as f:
        # first 4 byte is the number of nodes
		num_nodes = struct.unpack('I', f.read(4))[0]
		buffer = f.read()
		TSDFtup = struct.unpack('<{:d}f'.format(num_nodes), buffer)
		TSDF = np.array(TSDFtup, dtype=np.float32)
	
	return TSDF

def load_TSDF_allin1(path):
    with open(path, "rb") as f:
        # first 4 byte is the number of data
		num_data = struct.unpack('I', f.read(4))[0]
        # next 4 byte is the number of nodes
		num_nodes = struct.unpack('I', f.read(4))[0]
		buffer = f.read()
		TSDFtup = struct.unpack('<{:d}f'.format(num_nodes*num_data), buffer)
		TSDF = np.array(TSDFtup, dtype=np.float32).reshape(num_data, num_nodes)

    return TSDF


def saveTSDF_bin(TSDF, savepath):

	if not os.path.exists(os.path.dirname(savepath)):
		os.makedirs(os.path.dirname(savepath))

	with open(savepath, "wb") as f:
		num_nodes = TSDF.size

        # first 4 byte is the number of contained nodes, so when you read bin, skip first 4 byte.
		f.write(struct.pack("I", num_nodes))
		f.write(struct.pack('<{:d}f'.format(TSDF.size), *TSDF))



# Test code
def check_bin():
    # path_tsdf1 = "/home/onizuka/projects/tetrahedraregression/tools/test/meshes/132_15_c0018_TSDF.bin"
    # path_tsdf2 = "/home/onizuka/projects/tetrahedraregression/TSDF/TSDF_132_15_c0018_mesh.bin"
    # path_tsdf1 = "/home/onizuka/projects/tetrahedraregression/TSDF/crane_nu_test/clothedsmpl_centered0.bin"
    # path_tsdf1 = "/home/onizuka/projects/tetrahedraregression/TSDF/crane_nu_test/clothedsmpl_centered0_0.bin"
    # path_tsdf2 = "/home/onizuka/projects/tetrahedraregression/TSDF/crane_nu_test/clothedsmpl_centered0_nu=0.3.bin"
    # path_tsdf2 = "./tools/test/meshes/132_15_c0018_from_dimg.bin"
    path_tsdf1 = "/home/onizuka/projects/tetrahedraregression/TSDF/test/clothedsmpl_centered83.bin"
    path_tsdf2 = "/home/onizuka/projects/tetrahedraregression/TSDF/test/TSDF_Image3_0083_256x256.bin"


    tsdf1 = loadTSDF_bin(path_tsdf1)
    tsdf2 = loadTSDF_bin(path_tsdf2)

    m1 = np.mean(tsdf1)
    median1 = np.median(tsdf1)
    variance1 = np.var(tsdf1)
    stdev1 = np.std(tsdf1)

    m2 = np.mean(tsdf2)
    median2 = np.median(tsdf2)
    variance2 = np.var(tsdf2)
    stdev2 = np.std(tsdf2)
    print("----------------TSDF_fromM-----------------")
    print('Mean: {0:.2f}'.format(m1))
    print('Median: {0:.2f}'.format(median1))
    print('Variance: {0:.2f}'.format(variance1))
    print('StdDeviation: {0:.2f}'.format(stdev1))
    print('Max: {0:.2f}'.format(np.max(tsdf1)))
    print('Min: {0:.2f}'.format(np.min(tsdf1)))

    print("----------------TSDF_fromD-----------------")
    print('Mean: {0:.2f}'.format(m2))
    print('Median: {0:.2f}'.format(median2))
    print('Variance: {0:.2f}'.format(variance2))
    print('StdDeviation: {0:.2f}'.format(stdev2))
    print('Max: {0:.2f}'.format(np.max(tsdf2)))
    print('Min: {0:.2f}'.format(np.min(tsdf2)))

    tsdf1_woTvalues = tsdf1[(-32767<tsdf1) & (tsdf1<32767)]
    tsdf2_woTvalues = tsdf2[(-32767<tsdf2) & (tsdf2<32767)]
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    # ax3 = fig.add_subplot(121)
    # ax4 = fig.add_subplot(122)
    ax1.hist(x=tsdf1)
    ax2.hist(x=tsdf2)
    # ax1.hist(x=tsdf1_woTvalues)
    # ax2.hist(x=tsdf2_woTvalues)

    # ax1.set_title('Histgram TSDF_fromM')
    ax1.set_title('Histgram TSDF_1')
    # ax2.set_title('Histgram TSDF_fromD')
    ax2.set_title('Histgram TSDF_2')

    plt.show()

def load_pkl(parampath):
    with open(parampath, mode="rb") as f:
        pkl = pickle.load(f)
    return pkl

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


def colorize_cmesh_byTSDF():
    import matplotlib.pyplot as plt

    tetrapath = './models/TshapeCoarseTetraD.ply'
    # path_tsdf = "/home/onizuka/projects/tetrahedraregression/TSDF/TSDF_132_15_c0018_mesh.bin"
    # path_tsdf = "/home/onizuka/projects/tetrahedraregression/TSDF/crane_mu_test/clothedsmpl_centered0.bin"
    # path_tsdf = "/home/onizuka/projects/tetrahedraregression/tools/test/meshes/132_15_c0018_TSDF.bin"
    # path_tsdf = "./tools/test/meshes/132_15_c0018_from_dimg.bin"
    path_tsdf = "/home/onizuka/projects/tetrahedraregression/TSDF/crane_nu_test/clothedsmpl_centered0.bin"
    path_tsdf = "/home/onizuka/projects/tetrahedraregression/TSDF/test/TSDF_Image3_0083_256x256.bin"
    
    tsdf = loadTSDF_bin(path_tsdf)/32767.0
    c_mesh = pymesh.load_mesh(tetrapath)

    colormap = plt.get_cmap('jet')
    heatmap_threshold = 1
    RGB = colormap((tsdf+1)/2 / heatmap_threshold)[:,:3]*255
    save_ply("./tsdfdistribution.ply", c_mesh.vertices.T, f=c_mesh.faces.T, col=RGB.T, lab=None, fcol=None)


if __name__ == "__main__":

    coarse4 = 8211
    coarse3 = 16422
    coarse2 = 32844
    coarse1 = 65687
    coarse0 = 262745
    k=9

    savedir = "./randomconnection/"
    if not exists(savedir):
        os.makedirs(savedir)

    print("coarse 4 to 3: {} -> {}".format(coarse4, coarse3))
    adjlist = make_adjlist_random_connection(coarse4, coarse3, k, True)
    list2csv(adjlist, savedir + "/adjlist_4to3.csv")
    print("coarse 3 to 2: {} -> {}".format(coarse3, coarse2))
    adjlist = make_adjlist_random_connection(coarse3, coarse2, k, True)
    list2csv(adjlist, savedir + "/adjlist_3to2.csv")
    print("coarse 2 to 1: {} -> {}".format(coarse2, coarse1))
    adjlist = make_adjlist_random_connection(coarse2, coarse1, k, True)
    list2csv(adjlist, savedir + "/adjlist_2to1.csv")
    print("coarse 1 to original: {} -> {}".format(coarse1, coarse0))
    adjlist = make_adjlist_random_connection(coarse1, coarse0, k, True)
    list2csv(adjlist, savedir + "/adjlist_1to0.csv")

    check_bin()
    colorize_cmesh_byTSDF()