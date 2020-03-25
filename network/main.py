
from src import process
import sys
import tkinter as tk
from tkinter import filedialog
from os.path import splitext, basename, exists, isdir, isfile
import argparse
import cv2
import numpy as np
import glob

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser(description='Regress TSDF from single RGB image')

    parser.add_argument(
        '--logdir',
        default='./logs',
        type=str,
        help='Where logs and weights will be saved')
    parser.add_argument(
        '--datasetroot',
        default=None,
        type=str,
        help='Path to the directory contains dataset')

    parser.add_argument(
        '--imgdirs_train',
        default="./datasetpaths_train/imgdirs_train.txt",
        type=str,
        help='Relative path list to the imgs for training (from datasetroot)')
    parser.add_argument(
        '--tsdfdirs_train',
        default="./datasetpaths_train/TSDFdirs_train.txt",
        type=str,
        help='Relative path list to the TSDFs for training (from datasetroot)')

    parser.add_argument(
        '--imgdirs_test',
        default="./datasetpaths_test/imgdirs_test.txt",
        type=str,
        help='Relative path list to the imgs for testing (from datasetroot)')
    parser.add_argument(
        '--tsdfdirs_test',
        default="./datasetpaths_test/TSDFdirs_test.txt",
        type=str,
        help='Relative path list to the TSDFs for testing (from datasetroot)')
    parser.add_argument(
        '--paramdirs_test',
        default="./datasetpaths_test/paramdirs_test.txt",
        type=str,
        help='Relative path list to the SMPL params for testing (from datasetroot)')

    parser.add_argument(
        '--ids_train',
        default="./datasetpaths_train/Ids_train_arti.txt",
        # default=None,
        type=str,
        help="Specify the data to be used in training by ID (None -> use all loaded data for training)")
    parser.add_argument(
        '--ids_test',
        default="./datasetpaths_test/Ids_test_arti.txt",
        # default=None,
        type=str,
        help="Specify the data to be used in testing by ID (None -> use all loaded data for testing)")
        
    parser.add_argument(
        '--adjlists',
        default="./adjLists",
        type=str,
        help="Adjlists to construct PCN")
    parser.add_argument(
        '--imgpath_pred',
        default=None,
        type=str)
    parser.add_argument(
        '--mode',
        default=0,
        type=int,
        help="0:training, 1:prediction")
    parser.add_argument(
        '--lr',
        default=0.001,
        type=float)
    parser.add_argument(
        '--begin_epoch',
        default=0,
        type=int)
    parser.add_argument(
        '--end_epoch',
        default=150,
        type=int)
    parser.add_argument(
        '--batch_size',
        default=5,
        type=int)
    parser.add_argument(
        '--val_ratio',
        default=0.1,
        type=float)
    parser.add_argument(
        '--save_interval',
        default=50,
        type=int)
    parser.add_argument(
        '--pred_epoch',
        default=150,
        type=int)
    args = parser.parse_args()

    ############ config ############
    # Mode: Train=0, Predict=1
    mode = args.mode
    lr = args.lr
    begin_epoch = args.begin_epoch
    end_epoch = args.end_epoch
    batch_size = args.batch_size
    val_ratio = args.val_ratio
    save_interval = args.save_interval
    pred_epoch = args.pred_epoch
    logdir = args.logdir
    path_adjlists = args.adjlists
    datasetroot = args.datasetroot
    imgdirs_train = args.imgdirs_train
    tsdfdirs_train = args.tsdfdirs_train
    imgdirs_test = args.imgdirs_test
    tsdfdirs_test = args.tsdfdirs_test
    paramdirs_test = args.paramdirs_test
    ids_train = args.ids_train
    ids_test = args.ids_test
    ################################

    if mode == 0:
        print("Training mode")

        if datasetroot is None:
            print("Select dataset root")

            root = tk.Tk()
            root.withdraw()
            datasetroot = filedialog.askdirectory()
            root.destroy()
            if datasetroot == ():
                sys.exit()

        network = process.TetrahedraNetwork(path_adjlists)
        network.train(begin_epoch, end_epoch, batch_size, val_ratio, save_interval, lr, logdir, datasetroot, imgdirs_train, tsdfdirs_train, ids_train)


    elif mode == 1:
        print("Prediction mode")
        network = process.TetrahedraNetwork(path_adjlists)
        imgpath_pred = args.imgpath_pred
        if imgpath_pred is None:
            print("Select image(s)")
            root = tk.Tk()
            root.withdraw()
            imgpath_pred = filedialog.askopenfilenames()
            root.destroy()
            if imgpath_pred == () or imgpath_pred == "":
                sys.exit()
        elif isdir(imgpath_pred):
            imgpath_pred = sorted(glob.glob(imgpath_pred + "/*.png") + glob.glob(imgpath_pred + "/*.jpg"))
            imgpath_pred.sort(key=lambda x:len(x))
        elif isfile(imgpath_pred):
            imgpath_pred = [imgpath_pred]

        
        # Load input image
        Imgs = []
        savePaths = []
        for path in imgpath_pred:
            Imgs += [cv2.resize(cv2.imread(path, -1), (256, 256))[:,:,0:3]]
            savePaths += ["./result/{}_TSDF.bin".format(splitext(basename(path))[0])]
            
        network.tetranet.load_weights(logdir + '/weights_{0:d}.hdf5'.format(pred_epoch))
        network.predict(np.array(Imgs), savePaths)

    elif mode==2:
        print("Prepare data for evaluation")

        if datasetroot is None:
            print("Select dataset root")

            root = tk.Tk()
            root.withdraw()
            datasetroot = filedialog.askdirectory()
            root.destroy()
            if datasetroot == ():
                sys.exit()

        network = process.TetrahedraNetwork(path_adjlists)
        network.tetranet.load_weights(logdir + '/weights_{0:d}.hdf5'.format(pred_epoch))
        
        network.prepare_data(datasetroot, imgdirs_test, tsdfdirs_test, paramdirs_test, ids_test, savedir="./for_evaluation")