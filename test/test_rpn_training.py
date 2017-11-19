#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: test_rpn_training.py
# Author: Qian Ge <geqian1001@gmail.com>

from set_test_env import * 

import tensorflow as tf

from nets.faster_rcnn import RPN


if __name__ == '__main__':
	image = tf.placeholder(tf.float32, name='image',
                           shape=[1, None, None, 3])
    im_size = tf.placeholder(tf.float32, name='im_size',
                             shape=[1, 4])
    gt_bbox = tf.placeholder(tf.float32, name='gt_bbox',
                             shape=[1, None, 4])

    rpn = RPN(pre_train_path='', fine_tune=True)
    rpn.create_model([image, 1, im_size, gt_bbox])

    train_op = rpn.get_train_op()
    summery_op = rpn.summary_list

    db = DetectionDB('jpg', IM_PATH, XML_PATH, num_channel=3, rescale=600)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        batch_data = db.next_batch()
        im = batch_data[0]
        bbox = batch_data[1]
        re = sess.run(
        	[train_op, summery_op],
            feed_dict={image: im, im_size: [im.shape], gt_bbox: bbox})