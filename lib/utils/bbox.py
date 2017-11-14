#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: bbox.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np

# box [xmin, ymin, xmax, ymax]

def bbox_area(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

def bbox_intersec(box_1, box_2):
    
    box_2 = [box_2]
    box_2.append(box_1)
    box_2 = np.array(box_2)
    return np.append(np.amax(np.array(box_2[:, :2]), axis=0), 
                     np.amin(np.array(box_2[:, 2:]), axis=0))

def bbox_overlap_ratio(box_1, box_2):
    inter_area = bbox_area(bbox_intersec(box_1, box_2))
    return inter_area / (bbox_area(box_1) + bbox_area(box_2) - inter_area)

if __name__ == '__main__':
    box_1 = [1, 1, 300, 100]
    box_2 = [10, 10, 150, 200]

    bbox_overlap_ratio(box_1, box_2)