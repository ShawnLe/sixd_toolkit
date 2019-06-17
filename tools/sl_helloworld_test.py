__author__ = 'shawnle'
__email__ = 'letrungson1_at_gmail.com'

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

import json

# DATA_ROOT = r'D:\SL\PoseCNN\Loc_data\DUCK\POSE_iPBnet'
DATA_ROOT = r'D:\SL\PoseCNN\Loc_data\DUCK\POSE_iPBnet'
p0 = os.path.abspath(DATA_ROOT)

GEN_ROOT = r'D:\SL\Summer_2019\original_sixd_toolkit\sixd_toolkit\data\gen_data'

# model = inout.load_ply(r'D:\SL\Summer_2019\sixd_toolkit\data\sheep\textured.ply')
# model = inout.load_ply(r'D:\SL\Summer_2019\sixd_toolkit\data\ply\rotated.ply')
model = inout.load_ply(r'D:\SL\PoseCNN\Loc_data\DUCK\015_duck_toy\textured_m_text.ply')

# meta_file = os.path.join(p0, '{:06d}'.format(0) + '-meta.json')

# print('opening ', meta_file)
# with open(meta_file, 'r') as f:
#     meta_json = json.load(f)

# print('kyes ',meta_json.keys() )
# print('poses ')
# pose = np.array(meta_json['poses']).reshape(4,4) 
# print(pose)

# print('intrinsic_matrix ')
# print(np.array(meta_json['intrinsic_matrix']).reshape(3,3))

def flip_yz_image(image):

    ret = np.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype=image.dtype)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            ret[i,j,:] = image[image.shape[0]-i-1, image.shape[1]-j-1,:]

    return ret


for i in range(222):
    file_name = os.path.join(p0, '{:06d}'.format(i) + '-color.png')
    print(file_name)

    rgb = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    im_size = [rgb.shape[1], rgb.shape[0]]
    # cv2.imshow("rgb", rgb)
    # cv2.waitKey(1)

    # meta_file = os.path.join(p0, '{:06d}'.format(i) + '-meta.mat')
    # meta = scipy.io.loadmat(meta_file)

    meta_file = os.path.join(p0, '{:06d}'.format(i) + '-meta.json')
    print('opening ', meta_file)
    with open(meta_file, 'r') as f:
        meta = json.load(f)

    K = np.array(meta['intrinsic_matrix']).reshape(3,3)
    print('K',K)
    poses = np.array(meta['poses']).reshape(4,4)
    R = poses[:3,:3] 
    print ('R',R)
    t =  poses[:3,3]
    t /= 1000.
    # print('t',t)

    mdl_proj = renderer.render(model, im_size, K, R, t, mode='rgb', clip_near=.3, clip_far=6., shading='flat') 
    print("dtype", mdl_proj.dtype)
    print("max min", np.amax(mdl_proj), np.amin(mdl_proj))    

    # mdl_proj = flip_yz_image(mdl_proj)
    # cv2.imshow('model', mdl_proj)
    # cv2.waitKey(1)    
    mask_file = os.path.join(GEN_ROOT, '{:06d}-label.png'.format(i))
    cv2.imwrite(mask_file, mdl_proj)

    blend_name = os.path.join(GEN_ROOT, "{:06d}-blend.png".format(i))
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    mdl_proj_g = cv2.cvtColor(mdl_proj, cv2.COLOR_BGR2GRAY)
    alf = .5
    bet = 1 - alf
    bld = cv2.addWeighted(mdl_proj_g,alf,gray,bet,0.)
    cv2.imwrite(blend_name,bld)
    cv2.imshow('blend', bld)
    cv2.waitKey(1)    

