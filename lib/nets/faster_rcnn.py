#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: faster_rcnn.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf

from tensorcv.models.base import BaseModel
import tensorcv.models.layers as layers
from tensorcv.utils.common import apply_mask

from nets.vgg import VGG16_conv

# import sys
# sys.path.append('../../lib/')
import model.anchor as anchor
from model.losses import get_reg_loss, get_cls_loss
from model.bbox_anchor_transform import anchors_to_bbox
from model.layers import region_proposal_layer
from utils.viz import tf_draw_bounding_box
from utils.log import print_warning


class RPN(BaseModel):
    def __init__(self, pre_train_path, num_channels=3,
                 fine_tune=True, learning_rate=1e-5,
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

        self.set_model_input([self._image, self._keep_prob,
                              self._im_size, self._gt_bbox])
        self.set_dropout(self._keep_prob, keep_prob=0.5)
        self.set_train_placeholder([self._image, self._gt_bbox,
                                    self._im_size])

    def _create_model(self):
        input_im = self.model_input[0]
        keep_prob = self.model_input[1]

        with tf.name_scope('target_anchors'):
            im_size = self.model_input[2]
            gt_bbox = self.model_input[3]

            self._cls_mask, self._cls_label, self._targe_bbox_para,\
                self._pos_anchor_idx, self._pos_anchors, self._sampled_gt_bbox, all_anchors =\
                self._get_target_anchors(im_size[0], gt_bbox[0])

        self.layer = {}

        if self._vgg_path is None:
            is_load = False
            print_warning('No pre-train vgg loaded!')
        else:
            is_load = True
        vgg_model = VGG16_conv(is_load=is_load,
                               trainable_conv_3up=self._fine_tune,
                               pre_train_path=self._vgg_path)
        vgg_model.create_model([input_im, keep_prob])
        conv_out = vgg_model.layer['conv5_3']

        wd = 0.0005
        init_w = tf.random_normal_initializer(stddev=0.01)
        init_b = tf.random_normal_initializer(stddev=0.01)

        feat_map = layers.conv(conv_out, 3, 512, 'feat_map', wd=wd,
                               init_w=init_w, init_b=init_b)

        # cls layer
        with tf.name_scope('cls_layer'):
            cls = layers.conv(feat_map, 1, self._nanchor, 'cls_layer',
                          wd=wd, init_w=init_w, init_b=init_b)
            cls_prob = tf.sigmoid(cls, name='proposal_score')

        # reg layer
        with tf.variable_scope('reg_layer'):
            reg = []
            for i in range(0, 4):
                reg.append(layers.conv(feat_map, 1, self._nanchor,
                                       'reg_layer_{}'.format(i),
                                       wd=wd, init_w=init_w, init_b=init_b))

        with tf.variable_scope('rpn_reg_train'):
            pre_bbox_para_train = tf.transpose(
                tf.stack([apply_mask(c_reg[0], self._cls_label)
                          for c_reg in reg]))
            pre_proposal_bbox_train = tf.py_func(
                anchors_to_bbox,
                [self._pos_anchors, pre_bbox_para_train],
                tf.float64, name="pre_proposal_bbox_train")

        with tf.variable_scope('rpn_reg_predict'):
            proposal_pred_mask = tf.expand_dims(
                tf.where(tf.less(cls_prob, 0.5),
                         tf.zeros_like(cls_prob),
                         tf.ones_like(cls_prob)), dim=0)
            pre_proposal_para = tf.transpose(
                tf.stack([apply_mask(c_reg[0], proposal_pred_mask)
                          for c_reg in reg]))
            pre_proposal_bbox = tf.py_func(
                anchors_to_bbox,
                [all_anchors, pre_proposal_para],
                tf.float64, name="pre_proposal_bbox")




        # reg, pre_bbox_para, pre_proposal_bbx =
        #     region_proposal_layer(feat_map,
        #                           self._pos_anchors,
        #                           self._cls_label,
        #                           self._nanchor,
        #                           wd=wd, init_w=init_w, init_b=init_b,
        #                           name='reg_layer')

        self.layer['input'] = input_im
        self.layer['feat_map'] = feat_map
        self.layer['cls'] = cls
        self.layer['proposal_score'] = cls_prob
        self.layer['reg'] = reg
        self.layer['pre_bbox_para_train'] = pre_bbox_para_train
        self.layer['pre_proposal_bbox_train'] = pre_proposal_bbox_train

    def _get_target_anchors(self, im_size, gt_bbox):

        im_height = tf.to_int32(im_size[1])
        im_width = tf.to_int32(im_size[2])
        gt_bbox = tf.to_double(gt_bbox)

        pos_anchors, neg_anchors, pos_position, neg_position,\
            mask, label_map, sampled_gt_bbox, targe_bbox_para,\
            pos_anchor_idx, all_anchors =\
            tf.py_func(
                        anchor.anchor_training_samples,
                        [im_width, im_height, gt_bbox, self._stride,
                         self._ratio, self._scale,
                         self._pos_thr, self._neg_thr,
                         self._num_sample],
                        [tf.float64, tf.float64, tf.int64, tf.int64,
                         tf.int64, tf.int64, tf.float64, tf.float64,
                         tf.int64, tf.float64],
                        name="target_anchors")

        return tf.cast(mask, tf.int32), tf.cast(label_map, tf.int32),\
            targe_bbox_para, pos_anchor_idx, pos_anchors, sampled_gt_bbox, all_anchors

    def _get_loss(self):
        with tf.name_scope('loss'):
            cls_logits = self.layer['cls']
            cls_logits = tf.reshape(cls_logits, [tf.shape(cls_logits)[1],
                                                 tf.shape(cls_logits)[2],
                                                 tf.shape(cls_logits)[3]])

            self.cls_loss = get_cls_loss(cls_logits,
                                         self._cls_label,
                                         self._cls_mask)
            self.reg_loss = get_reg_loss(self.layer['reg'],
                                         self._pos_anchor_idx,
                                         self.layer['pre_bbox_para_train'],
                                         self._targe_bbox_para)

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
        self.test = self._get_loss()
        self.test_2 = self.test
        with tf.name_scope('scales'):
            summery_cls_label = tf.cast(self._cls_label, tf.float64)
            for i in range(0, self._nanchor):
                tf.summary.image('cls_{}'.format(i),
                                 tf.expand_dims(
                                    self.layer['cls'][:, :, :, i], dim=-1),
                                 collections=['train'])
                c_label = summery_cls_label[:, :, i]
                c_label = tf.reshape(
                    c_label,
                    [1, tf.shape(c_label)[0], tf.shape(c_label)[1], 1])
                tf.summary.image('targe_cls_{}'.format(i), c_label,
                                 collections=['train'])
        with tf.name_scope('bbox'):
            o_im = tf.cast(self.layer['input'], tf.uint8)
            category_index = {1: {'id': 1, 'name': 'non'},
                              2: {'id': 2, 'name': 'object'}}
            pre_prob = tf.gather(
                tf.reshape(self.layer['proposal_score'], [-1]),
                self._pos_anchor_idx)
            scores = tf.expand_dims(pre_prob, dim=0)
            classes = tf.expand_dims(
                tf.where(tf.less(pre_prob, 0.5),
                         tf.ones_like(pre_prob),
                         2 * tf.ones_like(pre_prob)), dim=0)

            pre_proposal_bbox_im = tf.where(
                tf.shape(self._pos_anchors)[0] == 0,
                o_im,
                tf_draw_bounding_box(o_im,
                                     self.layer['pre_proposal_bbox_train'],
                                     classes,
                                     scores,
                                     category_index,
                                     max_boxes_to_draw=200,
                                     min_score_thresh=0.01))
            tf.summary.image('pre_prosal', pre_proposal_bbox_im,
                             collections=['train'])

            scores = tf.expand_dims(tf.ones_like(self._pos_anchor_idx), dim=0)
            classes = tf.expand_dims(
                2 * tf.ones_like(self._pos_anchor_idx), dim=0)
            pos_anchor_im = tf.where(tf.shape(self._pos_anchors)[0] == 0,
                                     o_im,
                                     tf_draw_bounding_box(
                                        o_im,
                                        self._pos_anchors,
                                        classes,
                                        scores,
                                        category_index,
                                        max_boxes_to_draw=200,
                                        min_score_thresh=0.1))
            tf.summary.image('anchor', pos_anchor_im, collections=['train'])

            gt_bbox_im = tf_draw_bounding_box(o_im,
                                              self._sampled_gt_bbox,
                                              classes,
                                              scores,
                                              category_index,
                                              max_boxes_to_draw=200,
                                              min_score_thresh=0.1)
            tf.summary.image('gt', gt_bbox_im, collections=['train'])

        self.summary_list = tf.summary.merge_all('train')


SAVE_DIR = '/home/qge2/workspace/data/tmp/'

if __name__ == '__main__':
    from dataflow.detectiondb import DetectionDB
    vggpath = '/home/qge2/workspace/data/pretrain/vgg/vgg16.npy'
    im_path = '/home/qge2/workspace/data/dataset/VOCdevkit/VOC2007/JPEGImages/'
    xml_path = '/home/qge2/workspace/data/dataset/VOCdevkit/VOC2007/Annotations/'

    image = tf.placeholder(tf.float32, name='image',
                           shape=[1, None, None, 3])
    im_size = tf.placeholder(tf.float32, name='im_size',
                             shape=[1, 4])
    gt_bbox = tf.placeholder(tf.float32, name='gt_bbox',
                             shape=[1, None, 4])

    rpn = RPN(pre_train_path=vggpath, fine_tune=True)
    rpn.create_model([image, 1, im_size, gt_bbox])

    train_op = rpn.get_train_op()
    cls_cost_op = rpn.cls_loss
    reg_cost_op = rpn.reg_loss
    summery_op = rpn.summary_list

    test_op = rpn.test
    test_op_2 = rpn.test_2

    writer = tf.summary.FileWriter(SAVE_DIR)
    saver = tf.train.Saver()
    db = DetectionDB('jpg', im_dir=im_path, xml_dir=xml_path,
                     num_channel=3, rescale=600)

    trigger_step = 500
    faster_trigger_step = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        batch_data = db.next_batch()

        step_cnt = 1
        sum_cls_cost = 0
        sum_reg_cost = 0
        while db.epochs_completed < 10:
            batch_data = db.next_batch()
            im = batch_data[0]
            bbox = batch_data[1]
            re = sess.run([train_op, cls_cost_op, reg_cost_op,
                          summery_op, test_op, test_op_2],
                          feed_dict={image: im, im_size: [im.shape],
                                     gt_bbox: bbox})
            print('step: {}, cls_cost: {}, reg_cost: {}, cost: {}'.
                  format(step_cnt, re[1], re[2], re[1] + re[2]))
            sum_cls_cost += re[1]
            sum_reg_cost += re[2]

            if step_cnt % faster_trigger_step == 0:
                cls_cost_mean = sum_cls_cost * 1.0 / faster_trigger_step
                reg_cost_mean = sum_reg_cost * 1.0 / faster_trigger_step
                sum_cls_cost = 0
                sum_reg_cost = 0
                cost_mean = reg_cost_mean + cls_cost_mean

                s = tf.Summary()
                s.value.add(tag='mean/cls', simple_value=cls_cost_mean)
                s.value.add(tag='mean/reg', simple_value=reg_cost_mean)
                s.value.add(tag='mean/cost', simple_value=cost_mean)
                writer.add_summary(s, step_cnt)

            if step_cnt % trigger_step == 0:
                saver.save(sess, SAVE_DIR, global_step=step_cnt)
                writer.add_summary(re[3], step_cnt)
            step_cnt += 1
