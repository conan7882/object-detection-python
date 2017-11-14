#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Modified from https://github.com/belltailjp/selective_search_py/blob/master/features.py
# Similarity features of paper Selective Search for Object Recognition

import numpy as np
import math
# import skimage
# import skimage.filters
import scipy.ndimage.filters
# import matplotlib.pyplot as plt

COLOR_HIST_BIN = 25
ANGEL_BIN = 8
TEXTURE_BIN = 10


class Features(object):
    def __init__(self, image, label_im, n_region, mode=('C', 'T', 'S', 'F')):
        """

        Args:
            image (numpy.ndarray):
            label_im (numpy.ndarray):
            n_region (int): number of segmentation regions
        """
        self._image = image
        self._label_im = label_im
        self._n_region = n_region
        self._mode = mode

        self._imsize  = float(image.shape[0] * image.shape[1])

        self._box = self._get_bounding_box()
        self._size = self._get_size()

        if 'C' in mode:
            self._color_hist = self._get_color_hist()
        if 'T' in mode:
            self._texture_hist = self._get_texture_hist()
        print(mode)

    @ property
    def bounding_box(self):
        return self._box

    def _get_size(self):
        size_list = np.bincount(self._label_im.ravel(), 
                                minlength=self._n_region)
        return {i: size_list[i] for i in range(self._n_region)}

    def _get_bounding_box(self):
        box = {}
        for region in range(self._n_region):
            I, J = np.where(self._label_im == region)
            box[region] = (min(I), min(J), max(I), max(J))
        return {i: box[i] for i in range(self._n_region)}

    def _merge_bounding_box(self, i, j):
        box_i = self._box[i]
        box_j = self._box[j]
        box = ()
        for k in range(0, 2):
            box += min(box_i[k], box_j[k]),
        for k in range(2, 4):
            box += max(box_i[k], box_j[k]),
        return box

    def _merge_hist(self, hist_dict, i, j):
        hist_i = hist_dict[i]
        hist_j = hist_dict[j]

        return np.array([(self._size[i] * hist_i[k] + self._size[j] * hist_j[k]) / (self._size[i] + self._size[j])
                        for k in range(len(hist_i))])

    def _get_box_size(self, box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def _get_color_hist(self):
        # c_bin_width = int(math.ceil(255.0 / COLOR_HIST_BIN))
        bins = [range(self._n_region + 1), COLOR_HIST_BIN]
        r_hist = np.histogram2d(self._label_im.ravel(), self._image[:,:,0].ravel(), bins=bins)[0]
        g_hist = np.histogram2d(self._label_im.ravel(), self._image[:,:,1].ravel(), bins=bins)[0]
        b_hist = np.histogram2d(self._label_im.ravel(), self._image[:,:,2].ravel(), bins=bins)[0]

        hist = np.hstack([r_hist, g_hist, b_hist])
        l1_sum = np.sum(hist, axis = 1).reshape((self._n_region, 1))
        hist = np.nan_to_num(hist / l1_sum)

        return {i: hist[i] for i in range(self._n_region)}

    def _get_gradient_hist(self, gaussian):
        bins = [range(self._n_region + 1), TEXTURE_BIN]

        op = np.array([[-1, 0, 1]], dtype=np.float32)
        h = scipy.ndimage.filters.convolve(gaussian, op)
        hist_1 = np.histogram2d(self._label_im.ravel(), h.ravel(), bins=bins)[0]
        v = scipy.ndimage.filters.convolve(gaussian, op.transpose())
        hist_2 = np.histogram2d(self._label_im.ravel(), v.ravel(), bins=bins)[0]

        op = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], dtype=np.float32)
        h = scipy.ndimage.filters.convolve(gaussian, op)
        hist_3 = np.histogram2d(self._label_im.ravel(), h.ravel(), bins=bins)[0]
        v = scipy.ndimage.filters.convolve(gaussian, op.transpose())
        hist_4 = np.histogram2d(self._label_im.ravel(), v.ravel(), bins=bins)[0]

        hist = np.hstack([hist_1, hist_2, hist_3, hist_4])
        return hist

    def _get_texture_hist(self):
        # gaussian = skimage.filters.gaussian_filter(self._image, sigma=1.0, multichannel=True).astype(numpy.float32)
        gaussian = scipy.ndimage.filters.gaussian_filter(self._image.astype('float32'), sigma=1.0)
        r_hist = self._get_gradient_hist(gaussian[:, :, 0])
        g_hist = self._get_gradient_hist(gaussian[:, :, 1])
        b_hist = self._get_gradient_hist(gaussian[:, :, 2])

        hist = np.hstack([r_hist, g_hist, b_hist])
        l1_sum = np.sum(hist, axis = 1).reshape((self._n_region, 1))
        hist = np.nan_to_num(hist / l1_sum)

        return {i: hist[i] for i in range(self._n_region)}

    def _similarity_size(self, i, j):
        return 1 - (self._size[i] + self._size[j]) / self._imsize

    def _similarity_fill(self, i, j):
        box_merge = self._merge_bounding_box(i, j)
        return 1 - (self._get_box_size(box_merge) -
                    self._size[i] - self._size[j]) / self._imsize

    def _similarity_hist(self, hist_dict, i, j):
        hist_i = hist_dict[i]
        hist_j = hist_dict[j]

        return np.sum([min(hist_i[k], hist_j[k]) for k in range(len(hist_i))])

    def similarity(self, i, j):
        sim_sum = 0
        # color
        if 'C' in self._mode:
            s_color = self._similarity_hist(self._color_hist, i, j)
            sim_sum += s_color
        # texture
        if 'T' in self._mode:
            s_texture = self._similarity_hist(self._texture_hist, i, j)
            sim_sum += s_texture
        # size
        if 'S' in self._mode:
            s_size = self._similarity_size(i, j)
            sim_sum += s_size
        # fill
        if 'F' in self._mode:
            s_fill = self._similarity_fill(i, j)
            sim_sum += s_fill

        # print('color: {}, texture: {}, size: {}, fill: {}'.format(s_color, s_texture, s_size, s_fill))

        return sim_sum

    def merge_feature(self, i, j, new_id):
        merge_size = self._size[i] + self._size[j]
        merge_box = self._merge_bounding_box(i, j)
        self._size[new_id] = merge_size
        self._box[new_id] = merge_box

        if 'C' in self._mode:
            merge_color = self._merge_hist(self._color_hist, i, j)
            self._color_hist[new_id] = merge_color

        if 'T' in self._mode:
            merge_texture = self._merge_hist(self._texture_hist, i, j)
            self._texture_hist[new_id] = merge_texture
        
        
        


    


