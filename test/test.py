#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: test.py
# Author: Qian Ge <geqian1001@gmail.com>

from dataflow import DetectionDB

def test_dataflow():
	im_path = '../data/VOC2007/'
	xml_path = '../data/VOC2007/'
	db = DetectionDB('jpg', im_path, xml_path, num_channel=3, rescale=600)
	batch_data = db.next_batch()
	im = batch_data[0]
	bbox = batch_data[1]

if __name__ == '__main__':
	test_dataflow()
