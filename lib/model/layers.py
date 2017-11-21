#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: layers.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf

import tensorcv.models.layers as layers
from tensorcv.utils.common import apply_mask

from model.bbox_anchor_transform import anchors_to_bbox


def region_proposal_layer(feat_map, pos_anchors, cls_label, n_anchor_per_position,
                          wd=None, init_w=None, init_b=None,
                          name='reg_layer'):
    with tf.variable_scope(name):
        reg = []
        for i in range(0, 4):
            reg.append(layers.conv(feat_map, 1, n_anchor_per_position,
                                   'reg_layer_{}'.format(i),
                                   wd=wd, init_w=init_w, init_b=init_b))
        pre_bbox_para = tf.transpose(
            tf.stack([apply_mask(c_reg[0], cls_label)
                     for c_reg in reg]))
        pre_proposal_bbx = tf.py_func(
            anchors_to_bbox,
            [pos_anchors, pre_bbox_para],
            tf.float64, name="pre_proposal_bbx")

        return reg, pre_bbox_para, pre_proposal_bbx
