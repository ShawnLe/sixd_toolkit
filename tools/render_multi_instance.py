
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

K = cfg.INTRINSICS

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

rgb, dpt = renderer.render(model, im_size, K, Rs, ts)
