import cv2
import os
import numpy as np
import scipy.sparse as sp
import csv
import glob
import struct
import binascii
import time


def get_model_memory_usage(batch_size, model):
	from keras import backend as K

	shapes_mem_count = 0
	for l in model.layers:
		single_layer_mem = 1
		for s in l.output_shape:
			if s is None:
				continue
			single_layer_mem *= s
		shapes_mem_count += single_layer_mem

	trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
	non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

	number_size = 4.0
	if K.floatx() == 'float16':
		 number_size = 2.0
	if K.floatx() == 'float64':
		 number_size = 8.0

	total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
	gbytes = np.round(total_memory / (1024.0 ** 3), 3)
	return gbytes

def softmax(x, beta=100):
	c = np.max(beta*x)
	ex = np.exp(beta*x - c)
	sum_ex = np.sum(ex)
	return ex / sum_ex

def softargmax_3D(vol, beta=100):
	assert vol.ndim == 3, "x dim must be 2"
	
	s = softmax(vol, beta)
	x = np.arange(vol.shape[2])
	y = np.arange(vol.shape[1])
	z = np.arange(vol.shape[0])
	zz, yy, xx = np.meshgrid(z, y, x, indexing = 'ij')
	xmax = np.sum(s*xx)
	ymax = np.sum(s*yy)
	zmax = np.sum(s*zz)
	return zmax, ymax, xmax

def calcPose(heatmap3D, param, g_depth, use_softArgmax = False, is_orthogonal = False):
	camMat = np.reshape(np.array([[param['f'][0], 0, param['pp'][0]], [0, param['f'][1], param['pp'][1]], [0, 0, 1]]), (3, 3))
	camDist = np.array([param['k'][0], param['k'][1], param['p'][0], param['p'][1], param['k'][2]])

	points = []
	for i in range(heatmap3D.shape[3]):
		if use_softArgmax:
			# soft argmax implementation
			pos = softargmax_3D(heatmap3D[:,:,:,i], 100)
		else:
			#
			# argmax implementation
			#
			pos = np.array(np.unravel_index(heatmap3D[:,:,:,i].argmax(), heatmap3D[:,:,:,i].shape))

			# parabola fitting
			subPos = pos.astype('float32')

			R0 = heatmap3D[pos[0], pos[1], pos[2], i]
			if pos[0] > 0 and pos[0] < heatmap3D.shape[0] - 1:
				R_1 = heatmap3D[pos[0] - 1, pos[1], pos[2], i]
				R1 = heatmap3D[pos[0] + 1, pos[1], pos[2], i]
				subPos[0] += (R_1 - R1) / (2 * R_1 - 4 * R0 + 2 * R1)

			if pos[1] > 0 and pos[1] < heatmap3D.shape[1] - 1:
				R_1 = heatmap3D[pos[0], pos[1] - 1, pos[2], i]
				R1 = heatmap3D[pos[0], pos[1] + 1, pos[2], i]
				subPos[1] += (R_1 - R1) / (2 * R_1 - 4 * R0 + 2 * R1)

			if pos[2] > 0 and pos[2] < heatmap3D.shape[2] - 1:
				R_1 = heatmap3D[pos[0], pos[1], pos[2] - 1, i]
				R1 = heatmap3D[pos[0], pos[1], pos[2] + 1, i]
				subPos[2] += (R_1 - R1) / (2 * R_1 - 4 * R0 + 2 * R1)

			pos = subPos

		if is_orthogonal:
			points.append(np.array([pos[2], pos[1], pos[0]]))
		else:
			depth = pos[0] * 2 / heatmap3D.shape[0] - 1 + g_depth

			xy = pos[2:0:-1]
			xy = np.dot(np.linalg.inv(param['aug']), np.hstack((xy, np.array([1]))))[:2]
			xy = cv2.undistortPoints(np.reshape(np.array([xy]), (1, 1, 2)).astype('float32'), camMat, camDist, P = camMat)
			xy = (xy - param['pp']) * depth / param['f']
			points.append(np.array([xy[0,0,0], xy[0,0,1], depth]))

	return np.array(points)

def writePose(filename, pose, gt_pose = None, adjust_to_gt = True):
	numVertex = pose.shape[0]
	numEdge = numVertex - 1

	if gt_pose is not None:
		numVertex += gt_pose.shape[0]
		numEdge = numVertex - 2
		if adjust_to_gt:
			pose *= (np.max(gt_pose[:,1]) - np.min(gt_pose[:,1])) / (np.max(pose[:,1]) - np.min(pose[:,1]))
			pose += gt_pose[0] - pose[0]

	f = open(filename, 'w')
	f.write('ply\nformat ascii 1.0\n')
	f.write('element vertex ' + str(numVertex) + '\n')
	f.write('property float x\nproperty float y\nproperty float z\n')
	f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
	f.write('element edge ' + str(numEdge) + '\n')
	f.write('property int vertex1\nproperty int vertex2\n')
	f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
	f.write('end_header\n')

	for i in range(pose.shape[0]):
		f.write(str(pose[i,0]) + ' ' + str(pose[i,1]) + ' ' + str(pose[i,2]) + ' 255 0 0\n')

	if gt_pose is not None:
		for i in range(gt_pose.shape[0]):
			f.write(str(gt_pose[i,0]) + ' ' + str(gt_pose[i,1]) + ' ' + str(gt_pose[i,2]) + ' 0 255 0\n')

	# write edges
	for j in range(2 if gt_pose is not None else 1):
		for i in [(0, 1), (0, 4), (0, 7), (8, 10), (8, 13)]:
			f.write(str(i[0] + j * pose.shape[0]) + ' ' + str(i[1] + j * pose.shape[0]) + ' 255 255 255\n')
			f.write(str(i[1] + j * pose.shape[0]) + ' ' + str(i[1] + 1 + j * pose.shape[0]) + ' 255 255 255\n')
			f.write(str(i[1] + 1 + j * pose.shape[0]) + ' ' + str(i[1] + 2 + j * pose.shape[0]) + ' 255 255 255\n')

	f.close()


def load_adjLists(searchpath):
	print("Search path: ", searchpath)
	filePaths = reversed(sorted(glob.glob(searchpath)))
	adjLists = []
	for filepath in filePaths:
		print(filepath)
		with open(filepath) as csvf:
			reader = csv.reader(csvf)
			adjlist = []
			for line in reader:
				adjlist += [[int(s) for s in line]]
			adjLists += [adjlist]

	return adjLists

def make_adjmat(adjlist, shape, gen_identity=False):
	ilist = []
	jlist = []

	for i, adj in enumerate(adjlist):
		for j in adj:
			ilist += [i]
			jlist += [j]

	adjmat = sp.coo_matrix((np.ones(len(ilist)), (ilist, jlist)), shape=shape, dtype=np.float32).tocsr()

	# Generate identity matrix for self connection (use in GCN)
	if gen_identity:
		ilist = np.arange(len(adjlist))
		jlist = ilist
		adjmat_self = sp.coo_matrix((np.ones(len(ilist)), (ilist, jlist)), shape=shape, dtype=np.float32).tocsr()
		return [adjmat, adjmat_self]

	return [adjmat]




def make_adjMats(adjLists, shapelist, gen_identity=False):
	adjMats = []
	for adjlist, shape in zip(adjLists, shapelist):
		adjmat = make_adjmat(adjlist, shape, gen_identity)
		adjMats += [adjmat]

	return adjMats

def crop_imgs(Imgs, center_yx, shape_yx):
	cy = int(center_yx[0])
	cx = int(center_yx[1])
	y0 = cy-int(shape_yx[0]/2)
	x0 = cx-int(shape_yx[1]/2)
	y1 = y0 + int(shape_yx[0])
	x1 = x0 + int(shape_yx[1])
	return Imgs[:,y0:y1,x0:x1]

def loadTSDF_bin(filepath, num_nodes=159645):
	convVal = 32767.0

	with open(filepath, "rb") as f:
		try:
			num_nodes = struct.unpack('I', f.read(4))[0]
			buffer = f.read()
			TSDFtup = struct.unpack('<{:d}f'.format(num_nodes), buffer)
			TSDF = np.array(TSDFtup, dtype=np.float32) / convVal / 2.0 + 0.5
		except:
			print(filepath)
	return TSDF

def saveTSDF_bin(TSDF, savepath):
	convVal = 32767.0
	if not os.path.exists(os.path.dirname(savepath)):
		os.makedirs(os.path.dirname(savepath))

	TSDFout = (TSDF * 2.0 - 1.0) * convVal 
	with open(savepath, "wb") as f:
		num_nodes = TSDFout.size
		f.write(struct.pack("I", num_nodes))
		f.write(struct.pack('<{:d}f'.format(TSDFout.size), *TSDFout))

def saveTSDF_all_in_1(TSDFall, savepath):
	convVal = 32767.0
	if not os.path.exists(os.path.dirname(savepath)):
		os.makedirs(os.path.dirname(savepath))

	TSDFall_out = (TSDFall * 2.0 - 1.0) * convVal 
	with open(savepath, "wb") as f:
		num_data = TSDFall_out.shape[0]
		num_nodes = TSDFall_out.shape[1]
		print("num_data: {}".format(num_data))
		print("num_nodes: {}".format(num_nodes))

		TSDFall_out = TSDFall_out.flatten()
		f.write(struct.pack("I", num_data))
		f.write(struct.pack("I", num_nodes))
		f.write(struct.pack('<{:d}f'.format(TSDFall_out.size), *TSDFall_out))

def loadTSDF(filepath):
	convVal = 32767.0

	with open(filepath) as f:
		TSDFstr = f.read().split()
	TSDF = []
	for tsdfstr in TSDFstr:
		TSDF += [float(tsdfstr)/convVal]
	
	TSDF = np.array(TSDF, dtype=np.float32) / 2.0 + 0.5
	# print ("Max: ", np.max(self.TSDF))
	# print ("Min: ", np.min(self.TSDF))
	return TSDF
			
def saveTSDF(tsdf, savepath="result/TSDFpred.txt"):
	convVal = 32767.0
	print("shape: ", tsdf.shape)
	if not os.path.exists(os.path.dirname(savepath)):
		os.makedirs(os.path.dirname(savepath))

	tsdfout = (tsdf * 2.0 - 1.0) * convVal 
	print ("Max: ", np.max(tsdfout))
	print ("Min: ", np.min(tsdfout))
	with open(savepath, "w") as f:
		for i in range(len(tsdfout)):
			f.write(str(tsdfout[i]) + "\n")

if __name__ == "__main__":
	for i in range(12):
		path = "C:/Users/onizuka/Desktop/projects/tetrahedraregression/tetranet/DB/test_2camavg/TSDF_{:d}.txt".format(i)
		binpath = "C:/Users/onizuka/Desktop/projects/tetrahedraregression/tetranet/DB/test_2camavg/TSDF_{:d}.bin".format(i)

		start = time.time()
		TSDF = loadTSDF(path)
		print(time.time() - start)
		start = time.time()
		saveTSDF_bin(TSDF, binpath)
		print(time.time() - start)
		start = time.time()
		TSDFbin = loadTSDF_bin(binpath)
		print(time.time() - start)

		print(sum(TSDFbin-TSDF))
		print(TSDFbin[0:10])
