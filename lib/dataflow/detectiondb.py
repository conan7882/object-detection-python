#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: detectiondb.py
# Author: Qian Ge <geqian1001@gmail.com>

import xml.etree.ElementTree as ET
from tensorcv.dataflow.image import ImageFromFile

class DetectionDB(ImageFromFile):
    """ Base class of dataset for object detection (with bounding box)
    """
    def __init__(self):
        self._class_dict = {}
        self._nclass = 0


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
            box_list.append((cur_class, box, name))

        return box_list          

if __name__ == '__main__':
    db = DetectionDB()

    xml_path = '/Users/gq/workspace/Dataset/VOCdevkit/VOC2007/Annotations/000654.xml'
    db._parse_bbox_xml(xml_path)

