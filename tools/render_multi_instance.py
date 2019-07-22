
__author__ = 'shawnle'
__email__ = 'letrungson1@gmail.com'

import os
import sys
import glob
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import scipy.io

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pysixd import inout
from pysixd import misc
from pysixd import renderer

from pysixd import transform as tf

import cfg

rgb = cv2.imread(cfg.SAMPLE_RGB, cv2.IMREAD_UNCHANGED)

num_inst = cfg.NUM_INST

model = inout.load_ply(cfg.MESH_ROOT + '/textured.ply')

K = np.array(cfg.INTRINSICS).reshape(3,3)

im_size = cfg.IMAGE_SHAPE
im_size = (int(im_size[0]), int(im_size[1]))

gt_poses = cfg.GT_POSES
Rs = []
ts = []
for i in range(len(gt_poses)):
    RT = np.array(gt_poses[i]).reshape(4,4)
    R = RT[:3,:3]
    t = RT[:3,3] * .001 # to meter
    Rs.append(R)
    ts.append(t)

    print(R)
    print(t)

size = (im_size[1], im_size[0])
rgb, dpt, lbl = renderer.render(model, size, K, Rs, ts, mode='rgb+depth+label', clip_near=.3, clip_far=6., shading='flat')

np.save('lbl.npy', lbl)
print('lbl.npy is saved to disk.')

im_rescale_factor = cfg.IMG_RESCALE_FACTOR
rgb = cv2.resize(rgb, None, fx=im_rescale_factor, fy=im_rescale_factor, interpolation=cv2.INTER_LINEAR)
dpt = cv2.resize(dpt, None, fx=im_rescale_factor, fy=im_rescale_factor, interpolation=cv2.INTER_LINEAR)
lbl = cv2.resize(lbl, None, fx=im_rescale_factor, fy=im_rescale_factor, interpolation=cv2.INTER_LINEAR)

cv2.imshow('rgb', rgb)
cv2.imwrite('rgb.jpg', rgb)
cv2.imshow('dpt', dpt)
cv2.imwrite('dpt.jpg', dpt)
cv2.imshow('lbl', lbl*30)
cv2.imwrite('lbl.jpg', lbl*30)
cv2.waitKey()
