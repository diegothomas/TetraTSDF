import numpy as np
import chumpy as ch

import ply

import smpl_webuser
import smpl_webuser.serialization
import smpl_webuser.verts
import smpl_webuser.lbs
from smplify_public.code.render_model import render_model
from opendr.camera import ProjectPoints
import matplotlib.pyplot as plt
import cPickle as pickle
from argparse import ArgumentParser
from os.path import join, exists, abspath, dirname, basename, splitext



class Check_betas_byhand():

    def update_smpl(self):
        
        print (self.idx, self.betas)

        self.sv = smpl_webuser.verts.verts_decorated(
            trans = ch.array(self.trans),
            # pose = ch.array(self.smpl['pose']),
            # pose= ch.zeros(24*3),
            pose = ch.array(self.pose),
            v_template=self.model.v_template,
            J=self.model.J_regressor,
            betas=ch.array(self.betas),
            shapedirs=self.model.shapedirs[:, :, :self.model.shapedirs.shape[2]],
            weights=self.model.weights,
            kintree_table=self.model.kintree_table,
            bs_style=self.model.bs_style,
            f=self.model.f,
            bs_type=self.model.bs_type,
            posedirs=self.model.posedirs)

        self.img = (render_model(self.sv.r, self.model.f, self.w, self.h, self.cam, far=20) * 255.).astype('uint8')
        plt.clf()
        plt.imshow(self.img)
        plt.draw()
    

    def overwrite_betas(self, pklpath):
        params = np.load(pklpath)
        if "betas" in params:
            params["betas"] = self.betas
        else:
            params.setdefault("betas", self.betas)

        with open(pklpath, 'w') as outf:
            pickle.dump(params, outf)


    def run(self):
        self.save_count = 0
        def onKey(event):

            # print event.key
            if event.key == 'up':
                self.betas[self.idx] += 1
            elif event.key == 'down':
                self.betas[self.idx] -= 1
            elif event.key == 'right' and self.idx+1<len(self.betas):
                self.idx += 1
            elif event.key == 'left' and self.idx>0:
                self.idx -= 1
            elif event.key == "m":
                print "Saved ply to: ", self.outdir
                # ply.save_ply(self.outdir + "mesh_onbetas_{}.ply".format(self.save_count), self.sv.r.T, f=self.model.f.T)
                ply.save_ply(self.outdir + "J_onbetas_{}.ply".format(self.save_count), self.sv.J.T, f=None)
                self.save_count += 1

            elif event.key == "p" and self.pklpath_exists:
                self.overwrite_betas(self.pklpath)
                print ("Overwrote betas on: " + self.pklpath)
            elif event.key == "q":
                plt.close()
                return
            
            self.update_smpl()
        

        plt.connect('key_press_event',onKey)
        self.update_smpl()
        plt.show()
        

    def __init__(self, modelpath, outdir, pklpath=""):
        self.model = smpl_webuser.serialization.load_model(modelpath)

        self.flength = np.array([364.874, 364.874])
        self.center = np.array([259.965, 212.31])
        # self.rt = np.zeros(3)
        self.rt = np.array([-3.1415,  0,  0])
        # self.t = np.zeros(3)
        self.t = np.array([0,-0.3,2])
        self.w = 512
        self.h = 424
        self.cam = ProjectPoints(
            f=self.flength, rt=self.rt, t=self.t, k=np.zeros(5), c=self.center)


        self.idx = 0    
        self.pose = np.zeros(24*3)
        # self.pose[3:6] = np.array([0,  0,  0.5])
        # self.pose[6:9] = np.array([0,  0,  -0.5])
        self.betas = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0])
        self.trans = [0,0,0]

        self.outdir = outdir
        self.pklpath_exists = False
        self.pklpath = pklpath
        if not self.pklpath=="":
            self.pklpath_exists = True
            self.smpl = np.load(self.pklpath)
            self.pose = self.smpl["pose"]
            self.betas = self.smpl["betas"]
            self.trans = self.smpl["trans"]
            # self.trans = np.array([0,0,0])
            # self.trans = np.array([ 1.13615466,  1.25644733, -0.06495259])
            self.outdir = dirname(self.pklpath) + "/"

        
        
        self.Jdirs = np.dstack([self.model.J_regressor.dot(self.model.shapedirs[:, :, i])
                        for i in range(len(self.betas))])
        print(self.Jdirs)
        # J_onbetas = ch.array(self.Jdirs).dot(self.betas) + self.model.J_regressor.dot(self.model.v_template.r)
        # ply.save_ply(self.outdir + "J_onbetas_fromch.ply", J_onbetas.T, f=None)
        

        

if __name__ == "__main__":
    parser = ArgumentParser(description='pkl to smpl model')
    parser.add_argument(
        '--pklpath',
        type=str,
        default="",
        help='Path to smplparams')
    parser.add_argument(
        '--gender',
        type=str,
        default="male",
        help='Path to smplparams')
    args = parser.parse_args()

    if args.gender == "male":
        modelpath='./models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
        PATH_J_onbetas = "./models/J_onbetas_male/"
        print("Use male SMPL model")
    elif args.gender == "female":
        modelpath='./models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
        PATH_J_onbetas = "./models/J_onbetas_female/"
        print("Use female SMPL model")
    else:
        modelpath='./models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
        PATH_J_onbetas = "./models/J_onbetas_neutral/"
        print("Use neutral SMPL model")


    # main()
    cb = Check_betas_byhand(modelpath=modelpath, outdir=PATH_J_onbetas, pklpath=args.pklpath)
    cb.run()