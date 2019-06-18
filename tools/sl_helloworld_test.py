__author__ = 'shawnle'
__email__ = 'letrungson1_at_gmail.com'
__brief__ = 'read AP generated groundtruth poses and rgb and create *-label.png, *-depth.png'

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
# DATA_ROOT = r'D:\SL\PoseCNN\Loc_data\DUCK\POSE_iPBnet'
DATA_ROOT = '/media/shawnle/Data0/YCB_Video_Dataset/SLM_datasets/Exhibition/DUCK'
p0 = os.path.abspath(DATA_ROOT)

# GEN_ROOT = r'D:\SL\Summer_2019\original_sixd_toolkit\sixd_toolkit\data\gen_data'
GEN_ROOT = DATA_ROOT

# model = inout.load_ply(r'D:\SL\Summer_2019\sixd_toolkit\data\sheep\textured.ply')
# model = inout.load_ply(r'D:\SL\Summer_2019\sixd_toolkit\data\ply\rotated.ply')
# model = inout.load_ply(r'D:\SL\PoseCNN\Loc_data\DUCK\015_duck_toy\textured_m_text.ply')
# model = inout.load_ply('/media/shawnle/Data0/YCB_Video_Dataset/YCB_Video_Dataset/data_syn_LOV/models/015_duck_toy/textured_dense.ply')
model = inout.load_ply('/home/shawnle/Downloads/textured.ply')

print('model keys', model.keys())

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

# tuning pose
tx = -.001 # mm
ty = -.005 
tz = -.001
rz = 2./180.*math.pi # rad

xaxis, yaxis, zaxis = [1, 0, 0], [0, 1, 0], [0, 0, 1]
Tt = tf.translation_matrix([tx, ty, tz])
Rt = tf.rotation_matrix(rz, zaxis)

TT = np.eye(4)
TT[:3,:3] = Rt[:3,:3]
TT[:3,3] = Tt[:3,3]

# print('Tt = ')
# print(Tt)
# print('Rt = ')
# print(Rt)
print('TT = ')
print(TT)
# TT1 = np.dot(Tt,Rt)
# print('TT1 = ')
# print(TT1)

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
    # print('K',K)
    poses = np.array(meta['poses']).reshape(4,4)
    R = poses[:3,:3] 
    # print ('R',R)
    t =  poses[:3,3]
    t /= 1000.
    # print('t',t)

    # update with tuning
    Rt44 = np.eye(4)
    Rt44[:3,:3] = R
    Rt44[:3,3] = t
    Rt44 = np.dot(Rt44,TT)
    R = Rt44[:3,:3]
    t = Rt44[:3,3]

    mdl_proj, mdl_proj_depth = renderer.render(model, im_size, K, R, t, mode='rgb+depth', clip_near=.3, clip_far=6., shading='flat') 
    # print("dtype", mdl_proj.dtype)
    # print("max min", np.amax(mdl_proj), np.amin(mdl_proj))    

    # cv2.imshow('model', mdl_proj)
    # cv2.waitKey(1)    

    # depth format is int16
    # convert depth (see PCNN train_net.py)
    factor_depth = 10000
    zfar = 6.0
    znear = 0.25
    im_depth_raw = factor_depth * 2 * zfar * znear / (zfar + znear - (zfar - znear) * (2 * mdl_proj_depth - 1))
    I = np.where(mdl_proj_depth == 1)
    im_depth_raw[I[0], I[1]] = 0

    depth_file = os.path.join(GEN_ROOT, '{:06d}-depth.png'.format(i))
    cv2.imwrite(depth_file, im_depth_raw.astype(np.uint16))
    print('writing depth ' + depth_file)

    label_file = os.path.join(GEN_ROOT, '{:06d}-label.png'.format(i))
    # process the label image i.e. achieve nonzero pixel, then cast to cls_id value
    I = np.where(mdl_proj_depth > 0)
    # print('I shape',I.shape)
    label = np.zeros((rgb.shape[0], rgb.shape[1]))
    if len(I[0]) > 0:
        print('len I0',len(I[0]))
        print('label is exported')
        label[I[0],I[1]] = 1
    cv2.imwrite(label_file, label.astype(np.uint8))
    print('writing label ' + label_file)

    blend_name = os.path.join(GEN_ROOT, "{:06d}-blend.png".format(i))
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    mdl_proj_g = cv2.cvtColor(mdl_proj, cv2.COLOR_BGR2GRAY)
    alf = .5
    bet = 1 - alf
    bld = cv2.addWeighted(mdl_proj_g,alf,gray,bet,0.)
    cv2.imwrite(blend_name,bld)
    cv2.imshow('blend', bld)
    cv2.waitKey(1)    
    print('writing blend ' + blend_name)

    # revise pose json -> unit of pose is now in meter
    # save meta_data
    meta_file_rev = os.path.join(p0, '{:06d}'.format(i) + '-meta_rev.json')
    meta['poses'] = Rt44.flatten().tolist()
    with open(meta_file_rev, 'w') as fp:
        json.dump(meta, fp)
    print('writing meta ',meta_file_rev) 

