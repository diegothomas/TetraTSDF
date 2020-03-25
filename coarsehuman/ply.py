import sys, os
import skimage.color
import matplotlib.cm as cm

import numpy as np
# import file_io as fio


def load_ply(filename):
    if not os.path.isfile(filename):
        return None
    with open(filename, 'r') as fin:
        num_vertex = 0
        num_face = 0
        read_header = True
        while read_header:
            line = fin.readline()
            line_s = line.split(' ')
            if line_s[0] == 'element':
                if line_s[1] == 'vertex':
                    num_vertex = int(line_s[2])
                elif line_s[1] == 'face':
                    num_face = int(line_s[2])
            elif line_s[0].strip() == 'end_header':
                read_header = False
        vertices = np.ndarray((3, num_vertex))
        for i in range(num_vertex):
            line = fin.readline()
            line_s = line.split(' ')
            vertices[:, i] = [float(line_s[0]), float(line_s[1]), float(line_s[2])]
        faces = np.ndarray((3, num_face), dtype=np.int)
        for i in range(num_face):
            line = fin.readline()
            line_s = line.split(' ')
            faces[:, i] = [int(line_s[1]), int(line_s[2]), int(line_s[3])]
        return vertices, faces


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

    
def save_ply_edges(fname, m1, m2, c1, c2):
    n = len(c1)
    with open(fname, 'w') as fout:
        fout.write('ply\n')
        fout.write('format ascii 1.0\n')
        fout.write('element vertex %d\n' % (2*n))
        fout.write('property float x\n')
        fout.write('property float y\n')
        fout.write('property float z\n')
        fout.write('property uchar red\n')
        fout.write('property uchar green\n')
        fout.write('property uchar blue\n')
        fout.write('element edge %d\n' % n)
        fout.write('property int vertex1\n')
        fout.write('property int vertex2\n')
        fout.write('property uchar red\n')
        fout.write('property uchar green\n')
        fout.write('property uchar blue\n')
        fout.write('end_header\n')
        
        for i in range(n):
            v1 = m1[:,c1[i]]
            v2 = m2[:,c2[i]]
            col = np.array(cm.gist_rainbow(float(c1[i])/n))*255
            # fout.write('%f %f %f %.0f %.0f %.0f\n' % (v1[0], v1[1], v1[2], col[0], col[1], col[2]))
            # fout.write('%f %f %f %.0f %.0f %.0f\n' % (v2[0], v2[1], v2[2], col[0], col[1], col[2]))
            fout.write('%f %f %f %.0f %.0f %.0f\n' % (v1[0], v1[1], v1[2], 255, 0, 0))
            fout.write('%f %f %f %.0f %.0f %.0f\n' % (v2[0], v2[1], v2[2], 0, 255, 0))
    
        
        for i in range(n):
            fout.write('%d %d %.0f %.0f %.0f\n' % (2*i, 2*i+1, 0, 255, 0))


if __name__ == "__main__":
    pass
    # if len(sys.argv) < 3:
    #     print 'usage : python {0} [(input)obj file] [(output)ply file]'
    #     exit(0)

    # X, _, f = fio.load_obj_file(sys.argv[1])
    # save_ply(sys.argv[2], X, f=f)