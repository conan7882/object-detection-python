#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: faster_rcnn.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf
import imageio
import matplotlib.pyplot as plt

from tensorcv.models.base import BaseModel
import tensorcv.models.layers as layers
from tensorcv.utils.common import apply_mask

from vgg import VGG16_FCN

import sys
sys.path.append('../../lib/')
from dataflow.detectiondb import DetectionDB
import model.anchor as anchor
from model.masked_losses import masked_softmax_cross_entropy_with_logits



class RPN(BaseModel):
    def __init__(self, pre_train_path, num_channels=3, fine_tune=True, learning_rate=1e-3,
                 stride=16, ratios=(1, 0.5, 2), scales=(8, 16, 32),
                 pos_thr=0.7, neg_thr=0.3, num_sample_per_batch=128):
        self._vgg_path = pre_train_path
        self._fine_tune = fine_tune
        self._nchannel = num_channels

        self._nanchor = len(ratios) * len(scales)

        self._stride = stride
        self._ratio = ratios
        self._scale = scales
        self._pos_thr = pos_thr
        self._neg_thr = neg_thr
        self._num_sample = num_sample_per_batch

        self._lr = learning_rate

    def _create_input(self):
        self._keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self._image = tf.placeholder(tf.float32, name='image',
                            shape=[1, None, None, self._nchannel])
        self._gt_bbox = tf.placeholder(tf.float32, name='gt_bbox',
                            shape=[1, None, 4])
        self._im_size = tf.placeholder(tf.float32, name='im_size',
                            shape=[1, 4])


        self.set_model_input([self._image, self._keep_prob, self._im_size, self._gt_bbox])
        self.set_dropout(self._keep_prob, keep_prob=0.5)
        self.set_train_placeholder([self._image, self._gt_bbox, self._im_size])

    def _create_model(self):
        input_im = self.model_input[0]
        keep_prob = self.model_input[1]

        im_size = self.model_input[2]
        gt_bbox = self.model_input[3]
        self._get_target_anchors(im_size[0], gt_bbox[0])

        self.layer = {}

        vgg_model = VGG16_FCN(is_load=True, trainable_conv_3up=self._fine_tune,
                              pre_train_path=self._vgg_path)

        vgg_model.create_model([input_im, keep_prob])
        
        conv_out = vgg_model.layer['conv5_3']

        wd = 0.0005
        init_w = tf.random_normal_initializer(stddev = 0.01)
        init_b = tf.random_normal_initializer(stddev = 0.01)

        feat_map = layers.conv(conv_out, 3, 512, 'feat_map', wd=wd, init_w=init_w, init_b=init_b)
        # cls layer
        cls = layers.conv(feat_map, 1, self._nanchor, 'cls_layer', wd=wd, init_w=init_w, init_b=init_b)
        
        # reg layer
        # with tf.variable_scope('reg_layer'):
        #     reg = []
        #     for i in range(0, self._nanchor):
        #         reg.append(layers.conv(feat_map, 1, 4, 'reg_layer_{}'.format(i),
        #                                wd=wd, init_w=init_w, init_b=init_b))

        self.layer['input'] = input_im
        self.layer['feat_map'] = feat_map
        self.layer['cls'] = cls
        # self.layer['reg'] = reg
        self.layer['output'] = vgg_model.layer['gap_out']

        sum_cls_label = tf.cast(self._cls_label, tf.float64)
        for i in range(0, self._nanchor):
            tf.summary.image('cls_{}'.format(i), tf.expand_dims(cls[:, :, :, i], dim=-1),
                                                                collections=['train'])
            c_label = sum_cls_label[:, :, i]
            c_label = tf.reshape(c_label, [1, tf.shape(c_label)[0], tf.shape(c_label)[1], 1])
            tf.summary.image('targe_cls', c_label, collections=['train'])
        tf.summary.image('input_im', input_im, collections=['train'])

    def _get_target_anchors(self, im_size, gt_bbox):

        im_height = tf.to_int32(im_size[1])
        im_width = tf.to_int32(im_size[2])
        gt_bbox = tf.to_double(gt_bbox)

        pos_box, neg_box, pos_position, neg_position, mask, label_map, sampled_gt_bbox, t_s=\
            tf.py_func(anchor.anchor_training_samples, 
                       [im_width, im_height, gt_bbox, self._stride, 
                        self._ratio, self._scale,
                        self._pos_thr, self._neg_thr,
                        self._num_sample],
                       [tf.float64, tf.float64, tf.int64, tf.int64,
                        tf.int64, tf.int64, tf.float64, tf.float64], 
                       name="gt_boxes")

        self.test = pos_box
        self._cls_label = label_map
        self._cls_mask = mask


    def _comp_cls_loss(self, pre_logits, label, mask):
        with tf.name_scope('cls_loss'):
            cross_entropy = masked_softmax_cross_entropy_with_logits(
                logits=pre_logits, labels=label, mask=mask)
            cross_entropy_loss = tf.reduce_mean(cross_entropy, 
                                name='cross_entropy_cls')
            tf.add_to_collection('losses', cross_entropy_loss)

            tf.summary.scalar('cls_loss', cross_entropy_loss, 
                              collections = ['train'])
            self.cls_loss = cross_entropy_loss


    def _get_loss(self):
        with tf.name_scope('loss'):
            cls_logits = self.layer['cls']
            cls_logits = tf.reshape(cls_logits, [tf.shape(cls_logits)[1], 
                                                 tf.shape(cls_logits)[2], 
                                                 tf.shape(cls_logits)[3]])

            self._comp_cls_loss(cls_logits, self._cls_label, self._cls_mask)
            return tf.add_n(tf.get_collection('losses'), name='result') 

    def _get_optimizer(self):
        return tf.train.AdamOptimizer(beta1=0.9, 
                                      learning_rate=self._lr) 

    def get_train_op(self):
        grads = self.get_grads()
        opt = self.get_optimizer()
        train_op = opt.apply_gradients(grads, name='train')

        self._setup_summery()

        return train_op

    def _setup_summery(self):
        self.summary_list = tf.summary.merge_all('train')

SAVE_DIR = '/home/qge2/workspace/data/tmp/'

if __name__ == '__main__':
    import numpy as np
    from utils.viz import draw_bounding_box
    vggpath = '/home/qge2/workspace/data/pretrain/vgg/vgg16.npy'
    im_path = '/home/qge2/workspace/data/dataset/VOCdevkit/VOC2007/JPEGImages/'
    xml_path = '/home/qge2/workspace/data/dataset/VOCdevkit/VOC2007/Annotations/'

    

    image = tf.placeholder(tf.float32, name='image',
                            shape=[1, None, None, 3])
    im_size = tf.placeholder(tf.float32, name='im_size',
                            shape=[1, 4])
    gt_bbox = tf.placeholder(tf.float32, name='gt_bbox',
                            shape=[1, None, 4])


    rpn = RPN(pre_train_path=vggpath, fine_tune=False)
    rpn.create_model([image, 1, im_size, gt_bbox])

    # pre_op = tf.nn.top_k(tf.nn.softmax(rpn.layer['output']), 
    #                         k=5, sorted=True)

    train_op = rpn.get_train_op()
    cost_op = rpn.cls_loss
    summery_op = rpn.summary_list


    writer = tf.summary.FileWriter(SAVE_DIR)
    db = DetectionDB('jpg', im_dir=im_path, xml_dir=xml_path, num_channel=3,
                 rescale=600)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        batch_data = db.next_batch()

        step_cnt = 1
        while db.epochs_completed < 1:
            batch_data = db.next_batch()
            im = batch_data[0]
            bbox = batch_data[1]
            re = sess.run([train_op, cost_op, summery_op], feed_dict={image: im, im_size: [im.shape], gt_bbox: bbox})
            print('step: {}, cls_cost: {}'.format(step_cnt, re[1]))
            writer.add_summary(re[2], step_cnt)
            step_cnt += 1

        # writer.add_summary(summary, self.global_step)

    # print(np.array(test[0]).shape)
    # print(np.array(test[1]).shape)
    # # draw_bounding_box(np.squeeze(im), pre)
    # fig, ax = plt.subplots(1)

    # # Display the image
    # ax.imshow(test[0])

    # ax.axis('off')

    # plt.show()




