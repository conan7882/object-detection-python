#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: bbox.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np

# box [xmin, ymin, xmax, ymax]

def bbox_overlaps(boxes, query_boxes):
    
    nbox = boxes.shape[0]
    nqbox = query_boxes.shape[0]
    overlap_mat = np.zeros((nbox, nqbox), dtype='float32')
    for i, box in enumerate(boxes):
        box_area = bbox_area(box)
        for j, query_box in enumerate(query_boxes):
            q_box_area = bbox_area(query_box)
            i_area = bbox_area_intersec(box, query_box)
            overlap_mat[i, j] = i_area / (box_area + q_box_area - i_area)
    return overlap_mat

def bbox_area(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

def bbox_area_intersec(box_1, box_2):
    box_2 = [box_2]
    box_2.append(box_1)
    box_2 = np.array(box_2)
    ibox = np.append(np.amax(np.array(box_2[:, :2]), axis=0), 
                     np.amin(np.array(box_2[:, 2:]), axis=0))
    iw = ibox[2] - ibox[0] + 1
    ih = ibox[3] - ibox[1] + 1
    return 0 if iw < 0 or ih < 0 else iw*ih


def bbox_intersec(box_1, box_2):
    
    box_2 = [box_2]
    box_2.append(box_1)
    box_2 = np.array(box_2)
    return np.append(np.amax(np.array(box_2[:, :2]), axis=0), 
                     np.amin(np.array(box_2[:, 2:]), axis=0))

def bbox_overlap_ratio(box_1, box_2):
    inter_area = bbox_area(bbox_intersec(box_1, box_2))
    return inter_area / (bbox_area(box_1) + bbox_area(box_2) - inter_area)

def box2ctrwh(box):
    if len(np.shape(np.array(box))) == 1:
        box = [box]
    box = np.array(box)
    w = box[:, 2] - box[:, 0] + 1
    h = box[:, 3] - box[:, 1] + 1
    cx = box[:, 0] + 0.5 * (w - 1)
    cy = box[:, 1] + 0.5 * (h - 1)
    return cx, cy, w, h

def ctrwh2box(cx, cy, w, h):
    xmin = cx - 0.5 * (w - 1)
    ymin = cy - 0.5 * (h - 1)
    xmax = w - 1 + xmin
    ymax = h - 1 + ymin
    return np.vstack([xmin, ymin, xmax, ymax]).transpose()

if __name__ == '__main__':
    box_1 = [1, 1, 300, 100]
    box_2 = [10, 10, 150, 200]

    bbox_overlap_ratio(box_1, box_2)
