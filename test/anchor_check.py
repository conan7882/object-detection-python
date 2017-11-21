#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: anchor_check.py
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

import bbox_check as bbox
from bbox_anchor_transform_check import comp_regression_paras


def anchor_training_samples(im_width, im_height, gt_bbox,
                            stride=16,
                            ratios=(1, 0.5, 2), scales=(8, 16, 32),
                            pos_thr=0.7, neg_thr=0.3, num_sample=128):
    """
    Sample training anchors for region proposal networks as described in
    paper Faster R-CNN. Default parameters are the same as the paper.
    Positive samples are the anchors overlap groundtruth bounding box greater
    than pos_thr or the max overlapping anchor for a groundtruth box.
    Negative samples are the anchors overlap all the groundtruth box less
    than neg_thr. Training samples are randomly picked num_sample samples
    from positive and negative samples respectively. If there is less positive
    samples than num_sample, it will pad with
    negative ones.

    Args:
        im_width (int): width of original imagei
        im_height (int): height of original image
        gt_bbox (np.array): list of groundtruth bounding box with format
            [[xmin, ymin, xmax, ymax], ...]
        stride (int): stride of feature extraction networks
            (e.x. stride=16 for VGG)
        ratios (list): list of ratios used for anchors generation
        scales (list): list of scales used for anchors generation
        pos_thr (float): threshold for assigning the positive samples
        neg_thr (float): threshold for assigning the negative samples
        num_sample (int): number of training samples picked from positive
            and negative samples

    Returns:
        pos_anchor (np.array): list of positive training anchor samples with
            format [[xmin, ymin, xmax, ymax], ...]
        neg_anchor (np.array): list of negative training anchor samples with
            format [[xmin, ymin, xmax, ymax], ...]
        pos_position (np.array): list of position of positive samples in
            output of cls layer [x, y, scale]
        neg_position (np.array): list of position of negative samples in
            output of cls layer
        mask (np.array): mask of training samples corresponding to output
            of cls layer
        label_map (np.array): labels {1 (object), 0, (background)} of training
            samples on output of cls layer
        sampled_gt_bbox (np.array): corresponding groundtruth bounding box for
            each positive training samples
        t_s (np.array): bounding box regression parameters for each positive
            training anchors with format [[tx, ty, tw, th], ...]
        pos_anchor_idx (int):

    Note:
        All corresponding outputs maintain the same orders.
    """

    # if f_w is None or f_h is None:
    f_size = list(map(int, [np.floor(im_width / stride),
                            np.floor(im_height / stride)]))
    f_w = f_size[0]
    f_h = f_size[1]

    # Order based on [n_anchor, feat_with, feat_height]
    im_anchors, anchor_position = gen_im_anchors(
        f_w, f_h, stride=stride, ratios=ratios, scales=scales)

    valid_anchors, valid_positions =\
        remove_cross_boundary_anchors(im_width, im_height,
                                      im_anchors, anchor_position)

    pos_anchor, neg_anchor, pos_position, neg_position, pos_gt_idx =\
        get_gt_anchors(valid_anchors, valid_positions, gt_bbox,
                       pos_thr=pos_thr, neg_thr=neg_thr,
                       num_sample=num_sample)

    n_anchor_set = len(ratios) * len(scales)

    mask = np.empty([f_h, f_w, n_anchor_set])
    mask.fill(0)
    label_map = np.empty([f_h, f_w, n_anchor_set])
    label_map.fill(0)

    mask = _fill_map(mask, pos_position, 1)
    mask = _fill_map(mask, neg_position, 1)
    label_map = _fill_map(label_map, pos_position, 1)

    mask = mask.astype(int)
    label_map = label_map.astype(int)

    pos_anchor_idx = _map_position_to_index(label_map, pos_position)

    sampled_gt_bbox = gt_bbox[pos_gt_idx, :]

    # pos_anchor_idx = _map_position_to_index(label_map, pos_position)

    # Order based on [feat_with, feat_height, n_anchor]
    trans_idx = _index_awh2wha(pos_anchor_idx)
    pos_anchor_idx = pos_anchor_idx[trans_idx]
    pos_anchor = _awh2wha(pos_anchor, trans_idx)
    pos_position = _awh2wha(pos_position, trans_idx)
    sampled_gt_bbox = _awh2wha(sampled_gt_bbox, trans_idx)
    t_s = comp_regression_paras(pos_anchor, sampled_gt_bbox)

    print('number of samples: {}, number of positive: {}, {}, {}'.
          format(np.sum(mask), np.sum(label_map),
                 len(pos_gt_idx), len(pos_anchor_idx)))

    return pos_anchor, neg_anchor, pos_position, neg_position,\
        mask, label_map, sampled_gt_bbox, t_s, pos_anchor_idx, im_anchors


def _map_position_to_index(in_map, position):
    # print(in_map)
    t = position.transpose()
    # print(in_map.shape)
    t = np.vstack([t[1], t[0], t[2]])
    return np.ravel_multi_index(t, dims=in_map.shape)


def _index_awh2wha(awh_idx):
    return np.argsort(awh_idx)


def _awh2wha(in_array, idx):
    assert in_array.shape[0] == len(idx)
    return in_array[idx, :]


def _fill_map(map_fill, position, fill_val):
    # position [x, y]
    for p in position:
        # print(p)
        if (map_fill[p[1], p[0], p[2]] == 1):
            print('**{}'.format(p))
        map_fill[p[1], p[0], p[2]] = fill_val

    return map_fill


def get_gt_anchors(im_anchors, positions, gt_bbox,
                   pos_thr=0.7, neg_thr=0.3, num_sample=128):
    # TODO write bbox.bbox_overlaps in C++
    overlaps = bbox.bbox_overlaps(gt_bbox, im_anchors)
    ###
    max_overlap_gt = np.max(overlaps, axis=0)
    max_overlap_gt_idx = np.argmax(overlaps, axis=0)
    pos_idx = np.where(max_overlap_gt > pos_thr)
    pos_gt = max_overlap_gt_idx[pos_idx]

    max_overlap_anchor = np.max(overlaps, axis=1)
    max_overlap_anchor_id = np.argmax(overlaps, axis=1)
    gt_idx = np.array([i for i in range(len(gt_bbox))])
    gt_idx = np.delete(gt_idx, np.where(max_overlap_anchor < neg_thr))
    max_overlap_anchor_id = np.delete(
        max_overlap_anchor_id, np.where(max_overlap_anchor < neg_thr))

    pos_idx = np.concatenate((max_overlap_anchor_id, pos_idx[0]))
    pos_gt_idx = np.concatenate((gt_idx, pos_gt))

    pos_idx, unique_idx = np.unique(pos_idx, return_index=True)
    pos_gt_idx = pos_gt_idx[unique_idx]

    neg_idx = np.where(max_overlap_gt < neg_thr)[0]

    pos_anchor = im_anchors[pos_idx, :]
    pos_position = positions[pos_idx, :]

    neg_anchor = im_anchors[neg_idx, :]
    neg_position = positions[neg_idx, :]

    n_pos = pos_anchor.shape[0]
    n_neg = neg_anchor.shape[0]

    if n_neg < num_sample:
        raise GeneratorExit(
            'Not enough negtive anchor samples! (Should not happen.)')

    n_pos = min(n_pos, num_sample)
    n_neg = max(num_sample, 2 * num_sample - n_pos)

    pos_anchor, pos_position, pos_gt_idx = random_sample_anchor(
        pos_anchor, pos_position, n_pos, pos_idx, gt_idx=pos_gt_idx)

    neg_anchor, neg_position = random_sample_anchor(
        neg_anchor, neg_position, n_neg, neg_idx)

    return pos_anchor, neg_anchor, pos_position, neg_position, pos_gt_idx


def random_sample_anchor(anchors, positions, num_sample,
                         anchor_idx, gt_idx=None):
    n_anchor = anchors.shape[0]
    if n_anchor <= 0:
        # return anchors, positions, [], np.empty(shape=(0, 0), dtype=int)
        return anchors, positions, []
    r_idx = np.random.choice(n_anchor, size=num_sample, replace=False)
    if gt_idx is None:
        return anchors[r_idx, :], positions[r_idx, :]
    else:
        return anchors[r_idx, :], positions[r_idx, :], gt_idx[r_idx]


def remove_cross_boundary_anchors(im_w, im_h, anchors, positions):
    valid_idx = np.all([anchors[:, 0] >= 0, anchors[:, 1] >= 0,
                        anchors[:, 2] < im_w, anchors[:, 3] < im_h], axis=0)
    return anchors[valid_idx, :], positions[valid_idx]


def gen_im_anchors(width, height, stride=16,
                   ratios=(1, 0.5, 2), scales=(8, 16, 32)):
    """

    Args:
        width: width of feature map (output of the last conv layers)
        height: height of feature map
    """
    width = int(width)
    height = int(height)

    base_anchor_set = gen_anchors(stride=stride,
                                  ratios=ratios,
                                  scales=np.array(scales))
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
    anchor_position = np.vstack([shift_x.ravel() / stride,
                                 shift_y.ravel() / stride]).transpose()

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
