# Source work/joint_motion_20261119/render.py, SHA-256 104d910debed9c4bdd15a94bc251f5ba9604bc877d5775d3e4bcfd0382240a00

import case_settings as case

import cv2

import numpy as np

OUT=case.stage('joint_motion_20261119')

from observations import geometry, project, read_rgb

from fit_pivot import HERE, HANDS

class Recording:
    def __init__(self,g,native_file="native_wilor.npz"):
        self.group=g
        self.path,self.cal,self.rt,self.rows=geometry(g)
        with np.load(OUT/g/native_file) as z:
            self.iw=dict(zip(z['times'],range(len(z['times']))))
            self.native=z['joints_2d']
            self.w_valid=z['valid']
        self.labels=[]
        for hand in HANDS:
            with np.load(HERE/g/'contour/final'/(hand+'_pilot.npz')) as a,np.load(OUT/g/'final'/(hand+'_pilot.npz')) as b:
                self.labels.append((dict(zip(a['pair_indices'],range(len(a['pair_indices'])))),np.stack([a['joints'],b['joints']])))
        self.caps={}
    def read(self,t,c):
        if c not in self.caps:
            self.caps[c]=cv2.VideoCapture(str(self.path/f'cameras/camera{c}/video.mp4'),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2])
        return cv2.cvtColor(read_rgb(self.caps[c],int(self.rows[t][f'camera{c}_frame_index'])),cv2.COLOR_BGR2RGB)
    def poses(self,t,side,c):
        w=self.iw[t]
        native=self.native[w,c,side] if self.w_valid[w,c,side] else None
        look,xyz=self.labels[side]
        if t not in look:return [None,native,None,None]
        old,new=project(xyz[:,look[t]],self.rt[c],self.cal[c],side)
        return [None,native,old,new]
    def close(self):
        for cap in self.caps.values():cap.release()
