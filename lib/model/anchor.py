#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: anchor.py
# Author: Qian Ge <geqian1001@gmail.com>
# reference code [1]
# [1] https://github.com/endernewton/tf-faster-rcnn/blob/master/lib/layer_utils/generate_anchors.py

import numpy as np

def gen_im_anchors(weight, height, stride=16, ratios=(1, 0.5, 2), scales=(8, 16, 32)):
    base_anchor_set = gen_anchors(stride=stride, ratios=ratios, scales=scales)


def gen_anchors(stride=16, ratios=[1, 0.5, 2], scales=[8, 16, 32]):
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
    return np.transpose(np.vstack([xmin, ymin, xmax, ymax]))


if __name__ == '__main__':
    import time

    t = time.time()
    anchor = gen_anchors()
    print(time.time() - t)
    print(anchor)
