#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: test.py
# Author: Qian Ge <geqian1001@gmail.com>

from set_test_env import * 
from dataflow.detectiondb import DetectionDB

def test_dataflow():
    db = DetectionDB('jpg', IM_PATH, XML_PATH, num_channel=3, rescale=600)
    batch_data = db.next_batch()
    im = batch_data[0]
    bbox = batch_data[1]

if __name__ == '__main__':
    test_dataflow()
