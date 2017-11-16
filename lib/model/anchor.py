#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: anchor.py
# Author: Qian Ge <geqian1001@gmail.com>
# reference code [1]
# [1] https://github.com/endernewton/tf-faster-rcnn/blob/master/lib/layer_utils/generate_anchors.py

# output of gen_anchors() should be
# [[ -56.  -56.   71.   71.]
#  [-120. -120.  135.  135.]
#  [-248. -248.  263.  263.]
#  [ -84.  -40.   99.   55.]
#  [-176.  -88.  191.  103.]
#  [-360. -184.  375.  199.]
#  [ -36.  -80.   51.   95.]
#  [ -80. -168.   95.  183.]
#  [-168. -344.  183.  359.]]

import numpy as np
import numpy.matlib

import sys
sys.path.append('../../lib/')
import utils.bbox as bbox

def anchor_training_samples(im_width, im_height, gt_bbox, stride=16, 
                            ratios=(1, 0.5, 2), scales=(8, 16, 32),
                            pos_thr=0.7, neg_thr=0.3, num_sample=128):

    f_size = list(map(int, [np.round(im_width / stride), np.round(im_height / stride)]))
    f_w = f_size[0]
    f_h = f_size[1]

    im_anchors, anchor_position = gen_im_anchors(f_w, f_h, stride=stride, ratios=ratios, scales=scales)

    valid_anchors, valid_positions =\
        remove_cross_boundary_anchors(im_width, im_height, im_anchors, anchor_position)

    pos_box, neg_box, pos_position, neg_position =\
        get_gt_anchors(valid_anchors, valid_positions, gt_bbox,
                       pos_thr=pos_thr, neg_thr=neg_thr, num_sample=num_sample)

    mask = np.empty([f_h, f_w])
    mask.fill(0)
    label_map = np.empty([f_h, f_w])
    label_map.fill(0)

    mask = _fill_map(mask, pos_position, 1)
    mask = _fill_map(mask, neg_position, 1)

    label_map = _fill_map(label_map, pos_position, 1)
    # label_map = _fill_map(label_map, neg_position, 0)

    return pos_box, neg_box, pos_position, neg_position, mask, label_map

def _fill_map(map_fill, position, fill_val):
    # position [x, y]
    for p in position:
        map_fill[p[1], p[0]] = fill_val

    return map_fill


def get_gt_anchors(im_anchors, positions, gt_bbox, pos_thr=0.7, neg_thr=0.3, num_sample=128):
    # TODO write bbox.bbox_overlaps in C++
    overlaps = bbox.bbox_overlaps(gt_bbox, im_anchors)

    pos_idx = np.where(overlaps > pos_thr)

    max_idx = np.argmax(overlaps, axis=1)
    max_overlap = np.max(overlaps, axis=1)    
    max_idx = np.delete(max_idx, np.where(max_overlap < neg_thr))

    pos_idx = np.unique(np.concatenate((max_idx, pos_idx[1])))
    pos_box = im_anchors[pos_idx, :]
    pos_position = positions[pos_idx, :]

    neg_idx = np.where(overlaps < neg_thr)
    neg_box = im_anchors[neg_idx[1], :]
    neg_position = positions[neg_idx[1], :]

    n_pos = pos_box.shape[0]
    n_neg = neg_box.shape[0]

    if n_neg < num_sample:
        raise GeneratorExit('Not enough negtive anchor samples! (Should not happen.)')

    n_pos = min(n_pos, num_sample)
    n_neg = max(num_sample, 2 * num_sample - n_pos)

    pos_box, pos_position = random_sample_anchor(
        pos_box, pos_position, n_pos)

    neg_box, neg_position = random_sample_anchor(
        neg_box, neg_position, n_neg)

    return pos_box, neg_box, pos_position, neg_position


def random_sample_anchor(anchors, positions, num_sample):
    n_anchor = anchors.shape[0]
    r_idx = np.random.choice(n_anchor, size=num_sample, replace=False)
    return anchors[r_idx, :], positions[r_idx, :]



def remove_cross_boundary_anchors(im_w, im_h, anchors, positions):
    valid_idx = np.all([anchors[:, 0] >= 0, anchors[:, 1] >= 0, 
                anchors[:, 2] < im_w, anchors[:, 3] < im_h], axis=0)
    return anchors[valid_idx, :], positions[valid_idx]


def gen_im_anchors(width, height, stride=16, ratios=(1, 0.5, 2), scales=(8, 16, 32)):
    """

    Args:
        width: width of feature map (output of the last conv layers)
        height: height of feature map
    """
    width = int(width)
    height = int(height)

    base_anchor_set = gen_anchors(stride=stride, ratios=ratios, scales=np.array(scales))
    shift_y = list([y * stride for y in range(0, height)])
    shift_x = list([x * stride for x in range(0, width)])
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift_mesh = np.vstack([shift_x.ravel(), shift_y.ravel(), 
                            shift_x.ravel(), shift_y.ravel()]).transpose()

    nanchors = np.shape(base_anchor_set)[0]
    nshifts = np.shape(shift_mesh)[0]
    shift_anchors = np.reshape(base_anchor_set, (1, nanchors, 4)) +\
                    shift_mesh.reshape((1, nshifts, 4)).transpose((1, 0, 2))
    shift_anchors = np.reshape(shift_anchors, (nanchors * nshifts, 4))

    # Generate a list to record anchor postion in feature map and anchor id
    n_anchor = len(base_anchor_set)
    anchor_position = np.vstack([shift_x.ravel() / stride, shift_y.ravel() / stride]).transpose()

    anchor_position = np.array([list(map(int, (anchor[0], anchor[1], a_id)))
                       for anchor in anchor_position 
                       for a_id in range(0, n_anchor)])
    return shift_anchors, anchor_position


def gen_anchors(stride=16, ratios=[1, 0.5, 2], scales=2 ** np.arange(3, 6)):
    base_anchor = np.array([1, 1, stride, stride]) - 1
    anchors_ratio = gen_anchors_ratio(base_anchor, ratios)
    re_anchors = np.vstack([gen_anchors_scale(anchor, scales)
                            for anchor in anchors_ratio])

    return re_anchors


def gen_anchors_ratio(base_anchor, ratios):
    w, h, cx, cy = anchors2whcenter(base_anchor)
    size = w * h
    w_r = np.round((size / ratios) ** 0.5)
    h_r = np.round(w_r * ratios)
    return whcenter2anchors(w_r, h_r, cx, cy)


def gen_anchors_scale(base_anchor, scales):
    w, h, cx, cy = anchors2whcenter(base_anchor)
    w_r = w * scales
    h_r = h * scales
    return whcenter2anchors(w_r, h_r, cx, cy)


def anchors2whcenter(base_anchor):
    w = base_anchor[2] - base_anchor[0] + 1
    h = base_anchor[3] - base_anchor[1] + 1
    cx = base_anchor[0] + 0.5 * (w - 1)
    cy = base_anchor[1] + 0.5 * (h - 1)
    return w, h, cx, cy


def whcenter2anchors(w, h, cx, cy):
    xmin = cx - 0.5 * (w - 1)
    ymin = cy - 0.5 * (h - 1)
    xmax = w - 1 + xmin
    ymax = h - 1 + ymin
    return np.vstack([xmin, ymin, xmax, ymax]).transpose()


if __name__ == '__main__':
    import time
    from scipy import misc
    import imageio
    
    from dataflow.detectiondb import DetectionDB
    from utils.viz import draw_bounding_box

    t = time.time()
    
    # im_path = '/Users/gq/workspace/Dataset/VOCdevkit/VOC2007/JPEGImages/000654.jpg'
    # xml_path = '/Users/gq/workspace/Dataset/VOCdevkit/VOC2007/Annotations/000654.xml'

    im_path = '/home/qge2/workspace/data/dataset/VOCdevkit/VOC2007/JPEGImages/000654.jpg'
    xml_path = '/home/qge2/workspace/data/dataset/VOCdevkit/VOC2007/Annotations/000654.xml'

    stride = 8

    im = imageio.imread(im_path)
    im_h, im_w = im.shape[0], im.shape[1]
    f_w, f_h = np.round(im_w / stride), np.round(im_h / stride)

    im_anchors, anchor_position = gen_im_anchors(f_w, f_h, stride=stride)

    valid_anchors, valid_positions = remove_cross_boundary_anchors(im_w, im_h, im_anchors, anchor_position)
    # print(valid_anchors.shape[0])

    db = DetectionDB()
    gt_bbox = db._parse_bbox_xml(xml_path)
    pos_box, neg_box, pos_position, neg_position = get_gt_anchors(valid_anchors, valid_positions, gt_bbox, pos_thr=0.7, neg_thr=0.3)
    print(time.time() - t)
    draw_bounding_box(im, pos_box)

