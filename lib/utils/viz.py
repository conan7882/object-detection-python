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