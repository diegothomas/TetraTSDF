import os
import sys
from os.path import splitext, basename, exists
import numpy as np
import glob
import cv2
import math
from src import utils



class DataLoader():

    def save_Ids(self, Ids, savepath):
        with open(savepath, "w") as f:
            for idx in Ids:
                f.write("{}\n".format(idx))

    def load_Ids(self, filepath):
        Ids = []
        with open(filepath, "r") as f:
            lines = f.read().split()
            for idx in lines:
                Ids += [int(idx)]
        return np.array(Ids)

    def save_names(self, nameList, savepath):
        with open(savepath, "w") as f:
            for line in nameList:
                f.write(line + "\n")

    # def __init__(self, basedirpath, datasetlistname="folderlist_to_load.txt", batch_size=1, num_train=80, num_val=20, use_all=True, loadid=None, TSDFreldir="/TSDF", colorreldir="/imgs"):
    def __init__(self, datasetroot, imgdirlist, tsdfdirlist, batch_size, val_ratio, ids_train):

        self.val_ratio = val_ratio
        self.datasetroot = datasetroot
        self.imgdirlist = imgdirlist
        self.tsdfdirlist = tsdfdirlist
        self.batch_size = batch_size

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
        print("Read data from:")
        print(self.imgdirPaths)
        print(self.tsdfdirPaths)
        
        
        # Count number of all dataset
        self.countList = []
        self.nameList_color = []
        self.nameList_TSDF = []
        for imgdirpath, tsdfdirpath in zip(self.imgdirPaths, self.tsdfdirPaths):
            searchpath_color_png = imgdirpath + "/*.png"
            searchpath_color_jpg = imgdirpath + "/*.jpg"
            searchpath_TSDF = tsdfdirpath + "/*.bin"
            names_color = sorted(glob.glob(searchpath_color_png) + glob.glob(searchpath_color_jpg))
            names_color.sort(key=lambda x:len(x)) #String length and Dictionary sort
            names_TSDF = sorted(glob.glob(searchpath_TSDF))
            names_TSDF.sort(key=lambda x:len(x)) #String length and Dictionary sort

            if len(names_color) == len(names_TSDF):
                self.countList += [len(names_color)]
                self.nameList_color += names_color
                self.nameList_TSDF += names_TSDF
            else:
                print("The number of the input and target data is not same in:")
                print(imgdirpath, tsdfdirpath)
                self.countList += [0]
                print("color: {}, TSDF, {}".format(len(names_color), len(names_TSDF)))
        
        print("Num of available dataset: {0:d} (from {1:d} dir(s))".format(sum(self.countList), len(self.countList)))
        print(self.countList)

        # Generate index list
        if not ids_train is None:
            print("Select training data by loaded Ids")
            print("Path to Ids_train: {}".format(ids_train))
            self.Ids_all = self.load_Ids(ids_train)
            if len(self.Ids_all) > sum(self.countList):
                print("Invalid inputs")
                sys.exit()

        else:
            print("Use all available dataset ")
            self.Ids_all = np.random.choice(sum(self.countList), sum(self.countList), replace=False)

        self.num_all = len(self.Ids_all)
        self.num_val = int(self.num_all * self.val_ratio)
        self.num_train = self.num_all - self.num_val
        print("num_train: ", self.num_train)
        print("num_val: ", self.num_val)
        print("num_all: ", self.num_all)

        self.Ids_train = self.Ids_all[0:self.num_train]
        self.Ids_val  = self.Ids_all[self.num_train:self.num_all]
                
        self.steps_per_epoch_train = math.ceil(len(self.Ids_train) / self.batch_size)
        self.steps_per_epoch_val = math.ceil(len(self.Ids_val) / self.batch_size)

        # path_names_train = "./Names_train.txt"
        # path_names_val = "./Names_val.txt"
        # self.save_names(np.array(self.nameList_color)[self.Ids_train], path_names_train)
        # self.save_names(np.array(self.nameList_color)[self.Ids_val], path_names_val)

        # path_ids_train = self.basedirpath + "/Ids_train.txt"
        # path_ids_val = self.basedirpath + "/Ids_val.txt"
        # self.save_Ids(self.Ids_train, path_ids_train)
        # self.save_Ids(self.Ids_val, path_ids_val)


    
    def load_batch(self, usage="train"):

        if usage=="val":
            Ids = self.Ids_val
            steps_per_epoch = self.steps_per_epoch_val
        else:
            Ids = self.Ids_train
            steps_per_epoch = self.steps_per_epoch_train


        while True:
            start = 0
            end = start + self.batch_size
            

            for itr in range(steps_per_epoch) :
                # Generate batch
                Imgs = []
                TSDF = []

                for idx in Ids[start:end]:

                    # Load Img and TSDF
                    try:
                        # Imgs += [cv2.resize(cv2.imread(self.nameList_color[idx], -1)[:,:,0:3], (256, 256))]
                        Imgs += [cv2.imread(self.nameList_color[idx], -1)[:,:,0:3]]
                    except:
                        print("Got an error while reading {}".format(self.nameList_color[idx]))
                        sys.exit()
                    try:
                        TSDF += [utils.loadTSDF_bin(self.nameList_TSDF[idx])]
                    except:
                        print("Got an error while reading {}".format(self.nameList_TSDF[idx]))
                        sys.exit()

                # for i in range(len(Imgs)):
                #     cv2.imwrite("./val{}.png".format(i), Imgs[i])            
                Imgs = np.array(Imgs, dtype=np.float32)
                TSDF = np.array(TSDF, dtype=np.float32)
                yield (Imgs, TSDF)

                # Set next indices
                start = end
                end = end + self.batch_size
                if end > len(Ids):
                    end = len(Ids)

