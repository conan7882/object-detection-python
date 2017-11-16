#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: dataset.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import numpy as np
from scipy import misc

def get_file_list(file_dir, file_ext, sub_name=None):
    # assert file_ext in ['.mat', '.png', '.jpg', '.jpeg']
    re_list = []

    if sub_name is None:
        return np.array([(os.path.join(root, name), name)
            for root, dirs, files in os.walk(file_dir) 
            for name in files if name.lower().endswith(file_ext)]).transpose()
    else:
        return np.array([(os.path.join(root, name), name)
            for root, dirs, files in os.walk(file_dir) 
            for name in files if name.lower().endswith(file_ext) and sub_name.lower() in name.lower()]).transpose()



def load_image(im_path, read_channel=None, resize=None, rescale_shorter=None):
    if read_channel is None:
        im = misc.imread(im_path)
    elif read_channel == 3:
        im = misc.imread(im_path, mode='RGB')
    else:
        im = misc.imread(im_path, flatten=True)

    width = im.shape[0]
    height = im.shape[1]
    ratio = 1
    try:
        if width >= height:
            ratio = rescale_shorter / height
            n_h = rescale_shorter
            n_w = rescale_shorter / height * width
        else:
            ratio = rescale_shorter / width
            n_h = rescale_shorter / width * height
            n_w = rescale_shorter
        resize = [int(n_w), int(n_h)]
    except TypeError:
        pass

    
    if len(im.shape) < 3:
        try:
            im = misc.imresize(im, (resize[0], resize[1], 1))
        except TypeError:
            pass
        im = np.reshape(im, [1, im.shape[0], im.shape[1], 1])
    else:
        try:
            im = misc.imresize(im, (resize[0], resize[1], im.shape[2]))
        except TypeError:
            pass
        im = np.reshape(im, [1, im.shape[0], im.shape[1], im.shape[2]])
    return im, ratio
