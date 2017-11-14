#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: region_proposal_net.py
# Author: Qian Ge <geqian1001@gmail.com>

from tensorcv.models.base import BaseModel

from vgg import VGG16_FCN

def RPN(BaseModel):
    def __init__(self):
        pass

    def _create_input(self):
        pass

    def _create_model(self):
        if self._pre_train_path is None:
            is_load = False
        else:
            is_load = True
        vgg_model = VGG16_FCN(is_load=is_load, trainable=False,
                              pre_train_path=self._pre_train_path)