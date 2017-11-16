#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: faster_rcnn.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf
import imageio
from tensorcv.models.base import BaseModel
import tensorcv.models.layers as layers

from vgg import VGG16_FCN


class RPN(BaseModel):
    def __init__(self, pre_train_path, num_channels=3, fine_tune=True, num_anchors=9,
                 stride=16, ratios=(1, 0.5, 2), scales=(8, 16, 32),
                 pos_thr=0.7, neg_thr=0.3, num_sample_per_batch=128):
        self._vgg_path = pre_train_path
        self._fine_tune = fine_tune
        self._nchannel = num_channels
        self._nanchor = num_anchors

        self._stride = stride
        self._ratio = ratios
        self._scale = scales
        self._pos_thr = pos_thr
        self._neg_thr = neg_thr
        self._num_sample = num_sample_per_batch

    def _create_input(self):
        self._keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self._image = tf.placeholder(tf.float32, name='image',
                            shape=[None, None, None, self._nchannel])


        self.set_model_input([self._image, self._keep_prob])

    def _create_model(self):
        input_im = self.model_input[0]
        keep_prob = self.model_input[1]

        self.layer = {}

        vgg_model = VGG16_FCN(is_load=True, trainable_conv_3up=self._fine_tune,
                              pre_train_path=self._vgg_path)

        vgg_model.create_model([input_im, keep_prob])
        
        conv_out = vgg_model.layer['conv5_3']
        feat_map = layers.conv(conv_out, 3, 512, 'feat_map')
        # cls layer
        cls = layers.conv(feat_map, 1, 2, 'cls')
        # reg layer
        reg = []
        for i in range(0, self._nanchor):
            reg.append(layers.conv(feat_map, 1, 4, 'reg_{}'.format(i)))
        print(len(reg))

        self.layer['input'] = input_im
        self.layer['feat_map'] = feat_map
        self.layer['cls'] = cls
        self.layer['reg'] = reg
        self.layer['output'] = vgg_model.layer['gap_out']

    # def _get_target_anchors(self):

    #     anchor_training_samples(im_width, im_height, gt_bbox, stride=self._stride, 
    #                         ratios=self._ratio, scales=self._scale,
    #                         pos_thr=self._pos_thr, neg_thr=self._neg_thr,
    #                         num_sample=self._num_sample)

    def _comp_cls_loss(self, pre_logits, label):
        with tf.name_scope('cls_loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=pre_logits, labels=label)
            cross_entropy_loss = tf.reduce_mean(cross_entropy, 
                                name='cross_entropy_cls')
            tf.add_to_collection('losses', cross_entropy_loss)


    def _get_loss(self):
        with tf.name_scope('loss'):
            return tf.add_n(tf.get_collection('losses'), name='result')  


if __name__ == '__main__':
    vggpath = '/home/qge2/workspace/data/pretrain/vgg/vgg16.npy'
    im_path = '/home/qge2/workspace/data/dataset/VOCdevkit/VOC2007/JPEGImages/000654.jpg'
    # im_path = '/home/qge2/workspace/data/bird.jpg'
    im = imageio.imread(im_path)
    im = im.reshape((1, im.shape[0], im.shape[1], im.shape[2]))

    image = tf.placeholder(tf.float32, name='image',
                            shape=[None, None, None, 3])

    rpn = RPN(pre_train_path=vggpath, fine_tune=False)
    rpn.create_model([image, 1])

    pre_op = tf.nn.top_k(tf.nn.softmax(rpn.layer['output']), 
                            k=5, sorted=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        pre = sess.run(pre_op, feed_dict={image: im})
        print(pre)


