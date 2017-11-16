#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: masked_losses.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf

from tensorcv.utils.common import apply_mask

def masked_softmax_cross_entropy_with_logits(
        logits, labels, mask=None,
        name='masked_softmax_cross_entropy_with_logits'):
    """

    WARNING: This op expects unscaled logits, since it performs a softmax on
        logits internally for efficiency. Do not call this op with the output
        of softmax, as it will produce incorrect results.
    """
    with tf.name_scope(name):
        labels = tf.cast(labels, tf.int32)
        if not mask is None:
            mask = tf.cast(mask, tf.int32)
            logits = apply_mask(logits, mask)
            labels = apply_mask(labels, mask)

        return tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=labels, name='result') 