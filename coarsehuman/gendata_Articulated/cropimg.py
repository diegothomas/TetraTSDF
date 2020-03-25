import numpy as np
from skimage import io
from skimage.transform import resize
import json
import os
from os.path import dirname, basename, splitext, exists
import glob
import Tkinter as tk
import tkFileDialog
from argparse import ArgumentParser


def crop_img(img, center_xy, wh, outshape_wh, bgcolor_RGB=[0,127,0]):

    # Prepare base image
    img_after = np.empty([wh[1], wh[0], 3], dtype=np.uint8)
    img_after[:,:] = bgcolor_RGB

    # Calculate cropping window
    x0 = int(center_xy[0] - wh[0]/2)
    x1 = x0 + wh[0]
    y0 = int(center_xy[1] - wh[1]/2)
    y1 = y0 + wh[1]

    corr_x0 = 0
    corr_y0 = 0
    corr_x1 = wh[0]
    corr_y1 = wh[1]

    if x0<0:
        corr_x0 = -x0
        x0 = 0
    if y0<0:
        corr_y0 = -y0
        y0 = 0
    if x1>img.shape[1]:
        corr_x1 -= x1 - img.shape[1]
        x1 = img.shape[1]
    if y1>img.shape[0]:
        corr_y1 -= y1 - img.shape[0]
        y1 = img.shape[0]

    img_after[corr_y0:corr_y1, corr_x0:corr_x1] = img[y0:y1, x0:x1]

    return resize(img_after, outshape_wh)
    

if __name__ == "__main__":

    parser = ArgumentParser(description='Create mesh from TSDF')
    parser.add_argument(
        '--imgdir',
        type=str,
        default=None,
        help='Dir path to input images')
    parser.add_argument(
        '--skeletondir',
        type=str,
        default=None,
        help='Dir path to openpose skeletons .json')
    parser.add_argument(
        '--savedir',
        type=str,
        default=None,
        help='Dir path to save result mesh')
    parser.add_argument(
        '--outimgshape_xy',
        nargs="+",
        type=int,
        default=[256, 256],
        help="Width, height")
    parser.add_argument(
        '--bodyratio',
        type=float,
        default=0.65)
    args = parser.parse_args()


    imgdir = args.imgdir

    if imgdir is None:
        print("Select image directory")
        root = tk.Tk()
        root.withdraw()
        imgdir = tkFileDialog.askdirectory()
        root.destroy()
        if imgdir == ():
            exit()
    print("Image dir: " + imgdir)

    skeletondir = args.skeletondir
    if skeletondir is None:
        print("Select open pose skeleton directory (.json)")
        root = tk.Tk()
        root.withdraw()
        skeletondir = tkFileDialog.askdirectory()
        root.destroy()
        if skeletondir == ():
            exit()
    print("Skeleton dir: " + skeletondir)

    savedir = args.savedir
    if savedir is None:
        savedir = imgdir + "/cropped_imgs"
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    outimgshape_xy = args.outimgshape_xy
    bodyratio = args.bodyratio


    # Search paths to data
    imgPaths = sorted(glob.glob(imgdir + "/*.png"))
    if len(imgPaths) == 0:
        imgPaths = sorted(glob.glob(imgdir + "/*.jpg"))
    imgPaths.sort(key=lambda x:len(x)) #String length and Dictionary sort

    # skeletonPaths = sorted(glob.glob(skeletondir + "/*.json"))
    # skeletonPaths.sort(key=lambda x:len(x)) #String length and Dictionary sort


    from skimage.draw import (line, polygon, circle,
                          circle_perimeter,
                          ellipse, ellipse_perimeter,
                          bezier_curve)



    for imgpath in imgPaths:

        print("Img:      " + imgpath)
        
        skeletonpath = skeletondir + "/{}_keypoints.json".format(basename(splitext(imgpath)[0]))
        if not exists(skeletonpath):
            print("No skeleton. skip")
            continue

        print("Skeleton: " + skeletonpath)
        img = io.imread(imgpath)
        with open(skeletonpath, "r") as f:
            data = json.load(f)
        skeleton2d = np.array(data["people"][0]["pose_keypoints_2d"]).reshape(-1, 3)

        acc_mask = []
        for idx in range(len(skeleton2d)):
            if skeleton2d[idx, 2]>0.3:
                acc_mask += [idx]

        skeleton2d = skeleton2d[acc_mask]
        
        bbox_width = np.max(skeleton2d[:,0]) - np.min(skeleton2d[:,0])
        bbox_height = np.max(skeleton2d[:,1]) - np.min(skeleton2d[:,1])
        center_xy = [int(np.average(skeleton2d[:,0])), int(np.average(skeleton2d[:,1]))]
        # print(center_xy)

        if bbox_width < bbox_height:
            wh = [int(bbox_height/bodyratio), int(bbox_height/bodyratio)]
        else:
            wh = [int(bbox_width/bodyratio), int(bbox_width/bodyratio)]

        img_cropped = crop_img(img, center_xy, wh, outimgshape_xy)
        # img_cropped = crop_img(img, (img.shape[1]/2, img.shape[0]/2), [512, 512], [256, 256])
        # print(skeleton2d)
        # for coord in skeleton2d:
        #     rr, cc = circle_perimeter(int(coord[1]), int(coord[0]), 3)
        #     img[rr, cc, :] = (255, 255, 255)
        # rr, cc = circle_perimeter(center[0], center[1], 3)
        # img[rr, cc, :] = (255, 0, 0)
        # poly = np.array((
        #     (np.min(skeleton2d[:,1]), np.min(skeleton2d[:,0])),
        #     (np.min(skeleton2d[:,1]), np.max(skeleton2d[:,0])),
        #     (np.max(skeleton2d[:,1]), np.max(skeleton2d[:,0])),
        #     (np.max(skeleton2d[:,1]), np.min(skeleton2d[:,0])),
        # ))
        # rr, cc = polygon(poly[:, 0], poly[:, 1], img.shape)
        # img[rr, cc, 2] = 255

        # # Remove background
        # img_mask = np.where(img_cropped[:,:,1]>1.25*(img_cropped[:,:,2] + img_cropped[:,:,0]), 0, 1)
        # img_cropped *= np.dstack([img_mask,img_mask,img_mask])

        savepath = savedir + "/" + basename(splitext(imgpath)[0]) + "_{}x{}.png".format(outimgshape_xy[0], outimgshape_xy[1])
        # io.imsave(savepath, img)
        io.imsave(savepath, img_cropped)