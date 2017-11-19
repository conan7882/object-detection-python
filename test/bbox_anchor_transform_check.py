#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: bbox_anchor_transform.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np

import bbox_check as bbox


def comp_regression_paras(anchors, bboxes):
    """

    Args:
        anchors and bbox must have the same length
    Returns:
        [tx, ty, tw, th]
    """
    anchors = np.array(anchors)
    bboxes = np.array(bboxes)
    assert len(anchors) == len(bboxes)
    x, y, w, h = bbox.box2ctrwh(bboxes)
    xa, ya, wa, ha = bbox.box2ctrwh(anchors)

    tx = (x - xa) / wa
    ty = (y - ya) / ha
    tw = np.log(w / wa)
    th = np.log(h / ha)

    return np.vstack([tx, ty, tw, th]).transpose()


def bbox_to_anchors():
    pass


def anchors_to_bbox(anchors, t_para):
    # return [xmin, ymin, xmax, ymax]
    if len(np.shape(np.array(t_para))) == 1:
        t_para = [t_para]
    t_para = np.array(t_para)
    anchors = np.array(anchors)

    tx, ty, tw, th = t_para[:, 0], t_para[:, 1], t_para[:, 2], t_para[:, 3]
    xa, ya, wa, ha = bbox.box2ctrwh(anchors)

    assert len(xa) == len(tx)

    cx = tx * wa + xa
    cy = ty * ha + ya
    w = np.exp(tw) * wa
    h = np.exp(th) * ha

    return bbox.ctrwh2box(cx, cy, w, h)
