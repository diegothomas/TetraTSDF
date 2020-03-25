import os
from os.path import exists
import sys
import datetime
import time
import math
import cv2
import numpy as np
import glob
import shutil
from sklearn.metrics import mean_absolute_error, mean_squared_error

import keras
from keras.layers import *
from keras.layers.merge import _Merge
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
import keras.backend as K

import tensorflow as tf
import tensorboard as tb

from functools import partial
from src import utils
from src import tetranet
from src.dataloader import DataLoader

# Reduce memory usage
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
K.tensorflow_backend.set_session(session)



class TetrahedraNetwork():




	def __init__(self, path_adjlists):
		

		self.path_adjlists = path_adjlists

		self.img_shape = (256, 256, 3)



		# # Load adjLists and make adjMats for GCN
		# self.adjLists_forGCN = utils.load_adjLists(path_adjlists + "/adjlist_[0-9].csv")
		# if self.adjLists_forGCN == []:
		# 	print("No adjlists_forGCN are loaded")
		# 	return

		# shapelist = []
		# for i in range(len(self.adjLists_forGCN)):
		# 	size = len(self.adjLists_forGCN[i])
		# 	shapelist += [[size, size]]
		# self.adjMats_forGCN = utils.make_adjMats(self.adjLists_forGCN, shapelist, gen_identity=True)


		# Load adjLists for PCN
		self.adjLists_forPCN = utils.load_adjLists(path_adjlists + "/adjlist_[0-9]to[0-9].csv")
		if self.adjLists_forPCN == []:
			print("No adjlists_forPCN are loaded")
			return


		# Construct network
		# self.tetranet = tetranet.create_tetrahedra_network(self.adjLists_forPCN, self.adjMats_forGCN, shape = self.img_shape)
		self.tetranet = tetranet.create_tetrahedra_network(self.adjLists_forPCN, shape = self.img_shape)
		self.optimizer = Adam(lr = 0.005)
		# self.tetranet.summary()
		
		# self.tetranet.compile(loss='categorical_crossentropy',
		self.tetranet.compile(loss='mean_squared_error',
					  optimizer=self.optimizer,
					  metrics=["mean_squared_error"])


		# from keras.utils import plot_model
		# plot_model(self.tetranet, to_file='tetranet.png')	


	def step_decay(self, epoch):
		x = self.lr
		if epoch >= 50: x = 0.0001
		return x


	def train(self, begin_epoch, end_epoch, batch_size, val_ratio, save_interval, lr, logdir, datasetroot, imgdirlist, tsdfdirlist, ids_train):

		self.begin_epoch = begin_epoch
		self.end_epoch = end_epoch
		self.batch_size = batch_size
		self.val_ratio = val_ratio
		self.save_interval = save_interval
		self.lr = lr
		self.logdir = logdir
		self.datasetroot = datasetroot
		self.imgdirlist = imgdirlist
		self.tsdfdirlist = tsdfdirlist
		self.ids_train = ids_train
		
		# Generate logdir
		if not os.path.exists(self.logdir):
			os.makedirs(self.logdir)
		print("logdir: " + self.logdir)

		# Prepare dataset
		self.dl = DataLoader(self.datasetroot, self.imgdirlist, self.tsdfdirlist, self.batch_size, self.val_ratio, self.ids_train)

		# Check existance of logs
		if not os.path.exists(self.logdir + "/log.csv"):
			with open(self.logdir + "/log.csv", 'w') as f:
				f.write("{0:%Y-%m-%d %H:%M:%S}\n".format(datetime.datetime.now()))
		else:
			with open(self.logdir + "/log.csv", 'a') as f:
				f.write("{0:%Y-%m-%d %H:%M:%S}\n".format(datetime.datetime.now()))


		# Load weight when starting from intermediate epoch
		if(self.begin_epoch > 0):
			if(os.path.exists(self.logdir + "/weights_{0:d}.hdf5".format(self.begin_epoch))):
				print("Begin from " + self.logdir + "/weights_{0:d}.hdf5".format(self.begin_epoch))
				self.tetranet.load_weights(self.logdir + "/weights_{0:d}.hdf5".format(self.begin_epoch))
			else:
				print("File " + self.logdir + "/weights_{0:d}.hdf5".format(self.begin_epoch) + "does not exist")
				print("Start training from epoch 0")
				self.begin_epoch = 0


		# Start training
		start_time = datetime.datetime.now()
		generator_train = self.dl.load_batch("train")

		if self.dl.num_val > 0:
			generator_val = self.dl.load_batch("val")
			steps_per_epoch_val = self.dl.steps_per_epoch_val
		else:
			generator_val = None
			steps_per_epoch_val = None
			
		self.tetranet.fit_generator(generator_train, 
				steps_per_epoch=self.dl.steps_per_epoch_train, 
				initial_epoch=begin_epoch,
				epochs=end_epoch, 
				verbose=1, 
				callbacks=[keras.callbacks.CSVLogger(self.logdir + "/log.csv", separator=',', append=True), 
							keras.callbacks.ModelCheckpoint(self.logdir + "/weights_{epoch:d}.hdf5", period=save_interval),
							keras.callbacks.LearningRateScheduler(self.step_decay)],
				validation_data=generator_val, 
				validation_steps=steps_per_epoch_val,
				use_multiprocessing=True,
				workers=0,
				max_queue_size=5,
				shuffle=True)

		print("All processing time: ", datetime.datetime.now() - start_time)



	def predict(self, Imgs, savePaths):
		print("Predict")

		Imgs = np.array(Imgs, dtype=np.float32)
		out = self.tetranet.predict(Imgs)
		for i in range(len(out)):
			utils.saveTSDF_bin(out[i], savePaths[i])
			print("Saved result to: ", savePaths[i])

	
	def prepare_data(self, datasetroot, imgdirlist, tsdfdirlist, paramdirlist, ids_test, savedir):
		self.datasetroot = datasetroot
		self.imgdirlist = imgdirlist
		self.tsdfdirlist = tsdfdirlist
		self.paramdirlist = paramdirlist
		self.savedir = savedir

		# Parse .txt
		self.imgdirPaths = []
		with open(self.imgdirlist, "r") as f:
		    lines = f.read().split()
		    for imgdirname in lines:
		        if imgdirname[0] == "#":
		            continue
		        imgdirpath = self.datasetroot + "/" + imgdirname
		        if not exists(imgdirpath):
		            print("Dataset directory {} does not exists.".format(imgdirpath))
		        else: 
		            self.imgdirPaths += [imgdirpath]
		self.tsdfdirPaths = []
		with open(self.tsdfdirlist, "r") as f:
		    lines = f.read().split()
		    for tsdfdirname in lines:
		        if tsdfdirname[0] == "#":
		            continue
		        tsdfdirpath = self.datasetroot + "/" + tsdfdirname
		        if not exists(tsdfdirpath):
		            print("Dataset directory {} does not exists.".format(tsdfdirpath))
		        else: 
		            self.tsdfdirPaths += [tsdfdirpath]
		self.paramdirPaths = []
		with open(self.paramdirlist, "r") as f:
		    lines = f.read().split()
		    for paramdirname in lines:
		        if paramdirname[0] == "#":
		            continue
		        paramdirpath = self.datasetroot + "/" + paramdirname
		        if not exists(paramdirpath):
		            print("Dataset directory {} does not exists.".format(paramdirpath))
		        else: 
		            self.paramdirPaths += [paramdirpath]
		print("Read data from:")
		print(self.imgdirPaths)
		print(self.tsdfdirPaths)
		print(self.paramdirPaths)
		
		# Count number of all dataset
		self.countList = []
		self.nameList_color = []
		self.nameList_TSDF = []
		self.nameList_param = []
		for imgdirpath, tsdfdirpath, paramdirpath in zip(self.imgdirPaths, self.tsdfdirPaths, self.paramdirPaths):
			searchpath_color_png = imgdirpath + "/*.png"
			searchpath_color_jpg = imgdirpath + "/*.jpg"
			searchpath_TSDF = tsdfdirpath + "/*.bin"
			searchpath_param = paramdirpath + "/*.pkl"
			names_color = sorted(glob.glob(searchpath_color_png) + glob.glob(searchpath_color_jpg))
			names_color.sort(key=lambda x:len(x)) #String length and Dictionary sort
			names_TSDF = sorted(glob.glob(searchpath_TSDF))
			names_TSDF.sort(key=lambda x:len(x)) #String length and Dictionary sort
			names_param = sorted(glob.glob(searchpath_param))
			names_param.sort(key=lambda x:len(x)) #String length and Dictionary sort
		
			if len(names_color) == len(names_TSDF) == len(names_param):
				self.countList += [len(names_color)]
				self.nameList_color += names_color
				self.nameList_TSDF += names_TSDF
				self.nameList_param += names_param
			else:
				print("The number of the input and target data is not same in:")
				print(imgdirpath, tsdfdirpath, paramdirpath)
				self.countList += [0]
				print("color: {}, TSDF: {}, param: {}".format(len(names_color), len(names_TSDF), len(names_param)))
		
		print("Num of available dataset: {0:d} (from {1:d} dir(s))".format(sum(self.countList), len(self.countList)))
		print(self.countList)

		# Generate index list
		if not ids_test is None:
			print("Select training data by loaded Ids")
			print("Path to Ids_test: {}".format(ids_test))
			self.Ids_all = []
			with open(ids_test, "r") as f:
				lines = f.read().split()
				for idx in lines:
					self.Ids_all += [int(idx)]
			self.Ids_all = np.array(self.Ids_all)
			if len(self.Ids_all) > sum(self.countList):
				print("Invalid inputs")
				sys.exit()
			self.nameList_color = np.array(self.nameList_color)[self.Ids_all]
			self.nameList_TSDF = np.array(self.nameList_TSDF)[self.Ids_all]
			self.nameList_param = np.array(self.nameList_param)[self.Ids_all]

		# Copy all data
		if not exists(self.savedir + "/imgs"):
			os.makedirs(self.savedir + "/imgs")
		if not exists(self.savedir + "/TSDF_GT"):
			os.makedirs(self.savedir + "/TSDF_GT")
		if not exists(self.savedir + "/params"):
			os.makedirs(self.savedir + "/params")
		for i, (imgpath, TSDFpath, parampath) in enumerate(zip(self.nameList_color, self.nameList_TSDF, self.nameList_param)):
			shutil.copyfile(imgpath, self.savedir + "/imgs/{}.png".format(i))
			shutil.copyfile(TSDFpath, self.savedir + "/TSDF_GT/{}.bin".format(i))
			shutil.copyfile(parampath, self.savedir + "/params/{}.pkl".format(i))
		print("Saved imgs to: " + self.savedir + "/imgs")
		print("Saved TSDFs to: " + self.savedir + "/TSDF_GT")
		print("Saved params to: " + self.savedir + "/params")


		# Load images	
		print("Loading input imgs")
		Imgs = [cv2.resize(cv2.imread(imgpath, -1), (256, 256))[:,:,0:3] for imgpath in self.nameList_color]
		Imgs = np.array(Imgs, dtype=np.float32)
		print("Predict & save TSDFs...")
		out = self.tetranet.predict(Imgs)
		if not exists(self.savedir + "/TSDF_pred"):
			os.makedirs(self.savedir + "/TSDF_pred")
		for i in range(len(out)):
			utils.saveTSDF_bin(out[i], self.savedir + "/TSDF_pred/{}.bin".format(i))
		print("Saved predocted TSDFs to: {}".format(self.savedir + "/TSDF_pred"))


	