import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.io

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pysixd import inout
from pysixd import misc
from pysixd import renderer

DATA_ROOT = r'D:\SL\PoseCNN\Loc_data'
p0 = os.path.abspath(DATA_ROOT)

model = inout.load_ply(r'D:\SL\Summer_2019\sixd_toolkit\data\ply\sheep_meshlab.ply')
print('mdl keys', model.keys())
# print('model points', model['pts'])
print('model normals', model['normals'])
print('model colors', model['colors'])
print('model texture_uv', model['texture_uv'])
print('model faces', model['faces'])

for i in range(1):
    file_name = os.path.join(p0, '{:06d}'.format(i) + '-color.png')
    print(file_name)

    rgb = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    im_size = [rgb.shape[1], rgb.shape[0]]
    cv2.imshow("rgb", rgb)
    cv2.waitKey(0)

    meta_file =  os.path.join(p0, '{:06d}'.format(i) + '-meta.mat')
    meta = scipy.io.loadmat(meta_file)
    # print('meta keys', meta.keys())

    K = meta['intrinsic_matrix']
    print('K',K)
    poses = meta['poses']
    R = poses[:,:3]
    print ('R',R)
    t = poses[:,3]
    print('t',t)

    mdl_proj = renderer.render(model, im_size, K, R, t, mode='rgb', clip_near=0, clip_far=2000, shading='flat')
    print("dtype", mdl_proj.dtype)
    print("max min", np.amax(mdl_proj), np.amin(mdl_proj))    
    cv2.imshow('model', mdl_proj)
    cv2.waitKey(0)





