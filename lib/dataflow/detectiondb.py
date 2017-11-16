#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: detectiondb.py
# Author: Qian Ge <geqian1001@gmail.com>

import xml.etree.ElementTree as ET
import numpy as np
import os

from tensorcv.dataflow.base import RNGDataFlow
# from tensorcv.dataflow.normalization import identity
from tensorcv.dataflow.common import get_shape2D

import sys
sys.path.append('../../lib/')
from utils.dataset import get_file_list, load_image

class DetectionDB(RNGDataFlow):
    """ Base class of dataset for object detection (with bounding box)
    """
    def __init__(self, im_ext_name, im_dir='', xml_dir='',
                 num_channel=None,
                 shuffle=True, normalize=None,
                 resize=None, rescale=None):
        # self._im_ext = im_ext_name
        self._im_dir = im_dir
        self._xml_dir = xml_dir

        self._shuffle = shuffle

        self._class_dict = {}
        self._nclass = 0

        self.setup(epoch_val=0, batch_size=1)

        if num_channel is not None:
            self.num_channels = num_channel
            self._read_channel = num_channel
        else:
            self._read_channel = None

        self._resize = get_shape2D(resize)
        self._rescale = rescale

        self._load_file_list(im_ext_name)
        self._data_id = 0
        if self.size() == 0:
            print_warning('No {} files in folder {}'.\
                format(ext_name, data_dir))

    def next_batch(self):
        assert self._batch_size <= self.size(), \
        "batch_size cannot be larger than data size"

        start = self._data_id
        self._data_id += self._batch_size
        end = self._data_id

        if self._data_id + self._batch_size > self.size():
            self._epochs_completed += 1
            self._data_id = 0
            if self._shuffle:
                self._suffle_file_list()
        return self._load_data(start, end)

    def _load_data(self, start, end):
        im_list = []
        bbox_list = []
        for k in range(start, end):
            im_path = self._im_list[k]
            im_data = load_image(im_path, read_channel=self._read_channel,
                            resize=self._resize, rescale_shorter=self._rescale)
            im_list.extend(im_data[0])
            rescale_ratio = im_data[1]

            xml_path = self._xml_list[k]
            bbox = self._parse_bbox_xml(xml_path) * rescale_ratio
            bbox_list.append(bbox)

        return [im_list, bbox_list]


    def _load_file_list(self, ext_name):
        im_dir = os.path.join(self._im_dir)
        names = get_file_list(im_dir, ext_name)
        self._im_list = names[0]
        im_names = names[1]
        self._xml_list = np.array(['{}.xml'.format(os.path.join(
            self._xml_dir, im.split('.')[0])) for im in im_names])
    
        if self._shuffle:
            self._suffle_file_list()
        

    def _suffle_file_list(self):
        idxs = np.arange(self.size())
        self.rng.shuffle(idxs)

        self._im_list = self._im_list[idxs]
        self._xml_list = self._xml_list[idxs]


    def _parse_bbox_xml(self, xml_path):
        """

        Returns:
            [(class_id, [xmin, ymin, xmax, ymax], class_name)]
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        box_list = []

        for obj in root.findall('object'):
            name = obj.find('name').text

            try:
                cur_class = self._class_dict[name]
            except KeyError:
                cur_class = self._class_dict[name] = self._nclass
                self._nclass += 1

            box = obj.find('bndbox')
            xmin = float(box.find('xmin').text)
            ymin = float(box.find('ymin').text)
            xmax = float(box.find('xmax').text)
            ymax = float(box.find('ymax').text)
            box = [xmin, ymin, xmax, ymax]
            box_list.append(box)
            # box_list.append((cur_class, box, name))

        return np.array(box_list) 

    def size(self):
        return self._im_list.shape[0]   

if __name__ == '__main__':
    from utils.viz import draw_bounding_box

    im_path = '/Users/gq/workspace/Dataset/VOCdevkit/VOC2007/JPEGImages/'
    xml_path = '/Users/gq/workspace/Dataset/VOCdevkit/VOC2007/Annotations/'
    db = DetectionDB('jpg', im_path, xml_path, num_channel=3, rescale=600)

    batch_data = db.next_batch()
    im = batch_data[0]
    bbox = batch_data[1]
    print(bbox[0])
    draw_bounding_box(im[0], bbox[0])
    # xml_path = '/Users/gq/workspace/Dataset/VOCdevkit/VOC2007/Annotations/000654.xml'
    # bbox = db._parse_bbox_xml(xml_path)
    # print(bbox)


