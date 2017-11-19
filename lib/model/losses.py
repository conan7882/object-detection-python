#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: losses.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf

from tensorcv.utils.common import apply_mask


def get_reg_loss(pre_reg, pos_anchors, pre_t, t_s):
    with tf.name_scope('reg_loss'):
        t_s = tf.cast(t_s, pre_reg[0].dtype)
        reg_l1_smooth = l1_smooth_loss(pre_t, t_s)
        n_anchor_location = tf.cast(tf.shape(pre_reg[0])[1] *
                                    tf.shape(pre_reg[0])[2], tf.float32)
        reg_l1_smooth_loss = 10 * tf.reduce_sum(reg_l1_smooth) /\
            n_anchor_location
        tf.add_to_collection('losses', reg_l1_smooth_loss)
        return reg_l1_smooth_loss


def get_cls_loss(pre_logits, label, mask):
    with tf.name_scope('cls_loss'):
        cross_entropy = masked_sigmoid_cross_entropy_with_logits(
            logits=pre_logits, labels=label, mask=mask)
        cross_entropy_loss = tf.reduce_mean(cross_entropy,
                                            name='cross_entropy_cls')
        tf.add_to_collection('losses', cross_entropy_loss)
        return cross_entropy_loss


def masked_softmax_cross_entropy_with_logits(
        logits, labels, dim=-1, mask=None,
        name='masked_softmax_cross_entropy_with_logits'):
    """

    WARNING: This op expects unscaled logits, since it performs a softmax on
        logits internally for efficiency. Do not call this op with the output
        of softmax, as it will produce incorrect results.
    """
    with tf.name_scope(name):
        labels = tf.cast(labels, tf.int32)
        if mask is not None:
            mask = tf.cast(mask, tf.int32)
            logits = apply_mask(logits, mask)
            labels = apply_mask(labels, mask)

            # logits = tf.expand_dims(apply_mask(logits, mask), dim=-1)
            # labels = tf.expand_dims(apply_mask(labels, mask), dim=-1)

        return tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=labels, dim=dim, name='result')


def masked_sigmoid_cross_entropy_with_logits(
        logits, labels, mask=None,
        name='masked_sigmoid_cross_entropy_with_logits'):
    """

    """
    with tf.name_scope(name):
        labels = tf.cast(labels, tf.float32)
        if mask is not None:
            mask = tf.cast(mask, tf.int32)
            logits = apply_mask(logits, mask)
            labels = apply_mask(labels, mask)

        return tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits, labels=labels, name='result')


def l1_smooth_loss(pre, target, name='l1_smooth_loss'):
    with tf.name_scope(name):
        pre = tf.reshape(pre, [-1])
        target = tf.reshape(target, [-1])
        diff = tf.abs(tf.subtract(pre, target))
        return tf.where(diff < 1, tf.square(diff) * 0.5, diff - 0.5)
