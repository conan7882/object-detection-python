#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: test.py
# Author: Qian Ge <geqian1001@gmail.com>

from set_test_env import * 
import anchor_check as c_anchor
from dataflow.detectiondb import DetectionDB
import model.anchor as anchor 
import model.bbox_anchor_transform as ba_trans
import utils.bbox as bbox

def test_dataflow():
    db = DetectionDB('jpg', IM_PATH, XML_PATH, num_channel=3, rescale=600)
    batch_data = db.next_batch()
    im = batch_data[0]
    bbox = batch_data[1]
    return im, bbox


def test_bbox(box):
    cx, cy, w, h = bbox.box2ctrwh(box)
    r_box = bbox.ctrwh2box(cx, cy, w, h)
    box = box.astype(int)
    r_box = r_box.astype(int)
    for (b, r_b) in zip(box, r_box):
        assert np.array_equal(b, r_b), print(b, r_b)


def test_anchor_bbox_transform(anchors, bboxes):
    t_para = ba_trans.comp_regression_paras(anchors, bboxes)
    r_bboxes = ba_trans.anchors_to_bbox(anchors, t_para)
    bboxes = bboxes.astype(int)
    r_bboxes = r_bboxes.astype(int)
    assert np.array_equal(bboxes, r_bboxes), print(bboxes, r_bboxes)


def test_anchor(im, gt_bbox):
    # random.seed(a=1)
    im_w = im.shape[0]
    im_h = im.shape[1]
    re = anchor.anchor_training_samples(im_w, im_h, gt_bbox,
                                        stride=16,
                                        ratios=(1, 0.5, 2),
                                        scales=(8, 16, 32),
                                        pos_thr=0.6, neg_thr=0.3,
                                        num_sample=64)
    c_re = c_anchor.anchor_training_samples(im_w, im_h, gt_bbox,
                                            stride=16,
                                            ratios=(1, 0.5, 2),
                                            scales=(8, 16, 32),
                                            pos_thr=0.6, neg_thr=0.3,
                                            num_sample=64)

    checklist = [0, 2, 5, 6, 7, 9]

    for i in checklist:
        assert np.array_equal(re[i], c_re[i]), print(i)
    return c_re[0], c_re[6]

if __name__ == '__main__':
    im, gt_bbox = test_dataflow()
    test_bbox(gt_bbox[0])
    anchors, gt_bbox = test_anchor(im[0], gt_bbox[0])
    test_anchor_bbox_transform(anchors, gt_bbox)
