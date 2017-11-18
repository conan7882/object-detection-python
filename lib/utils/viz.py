#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: viz.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_bounding_box(im, box):
    box = np.array(box)
    if len(box.shape) == 1:
        box = [box]

    im = np.array(im, dtype=np.uint8)

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)

    # Create a Rectangle patch
    for c_box in box:
        rect = patches.Rectangle((c_box[0], c_box[1]), c_box[2] - c_box[0], c_box[3] - c_box[1],
                                  linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

    ax.axis('off')

    plt.show()


# def tf_draw_bounding_box(im, box):
#     """
#     Args:
#         box: [xmin, ymin, xmax, ymax]
#     """
#     import tensorflow as tf
#     # change box to [y_min, x_min, y_max, x_max]
#     im = tf.cast(im, tf.float32)
#     box = tf.cast(box, tf.float32)

#     im_h = tf.cast(tf.shape(im)[1], tf.float32)
#     im_w = tf.cast(tf.shape(im)[2], tf.float32)

#     box = tf.stack([box[:, 1] / im_h, box[:, 0] / im_w, box[:, 3] / im_h, box[:, 2] / im_w], axis=1)
#     box = tf.expand_dims(box, dim=0)
#     # return box

#     return tf.cast(tf.image.draw_bounding_boxes(im, box), tf.uint8)


def tf_draw_bounding_box(im, box, classes, scores, category_index,
                         max_boxes_to_draw=20, min_score_thresh=0.2):

    import tensorflow as tf
    from utils.visualization_utils import draw_bounding_boxes_on_image_tensors


    im = tf.cast(im, tf.uint8)
    box = tf.cast(box, tf.float32)
    classes = tf.cast(classes, tf.int32)

    # change box to [y_min, x_min, y_max, x_max]
    im_h = tf.cast(tf.shape(im)[1], tf.float32)
    im_w = tf.cast(tf.shape(im)[2], tf.float32)

    box = tf.stack([box[:, 1] / im_h, box[:, 0] / im_w, box[:, 3] / im_h, box[:, 2] / im_w], axis=1)
    box = tf.expand_dims(box, dim=0)

    bbox_im = draw_bounding_boxes_on_image_tensors(
            im,
            box,
            classes,
            scores,
            category_index,
            max_boxes_to_draw=max_boxes_to_draw,
            min_score_thresh=min_score_thresh)

    return bbox_im




