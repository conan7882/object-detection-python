import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def find_adjacent_region(label_im, label_id):
    x = label_im.astype('int32')
    region = label_id   # number of region whose neighbors we want

    y = x == region  # convert to Boolean

    rolled = np.roll(y, 1, axis=0)          # shift down
    rolled[0, :] = False             
    z = np.logical_or(y, rolled)

    rolled = np.roll(y, -1, axis=0)         # shift up 
    rolled[-1, :] = False
    z = np.logical_or(z, rolled)

    rolled = np.roll(y, 1, axis=1)          # shift right
    rolled[:, 0] = False
    z = np.logical_or(z, rolled)

    rolled = np.roll(y, -1, axis=1)         # shift left
    rolled[:, -1] = False
    z = np.logical_or(z, rolled)

    neighbors = set(np.unique(np.extract(z, x))) - set([region])
    return neighbors

def find_all_adjacent_region(x):
    n = x.max()
    tmp = np.zeros((n+1, n+1), bool)

    # check the vertical adjacency
    a, b = x[:-1, :], x[1:, :]
    tmp[a[a!=b], b[a!=b]] = True

    # check the horizontal adjacency
    a, b = x[:, :-1], x[:, 1:]
    tmp[a[a!=b], b[a!=b]] = True
    # register adjacency in both directions (up, down) and (left,right)
    result = (tmp | tmp.T)
    result.astype(int)
    return np.column_stack(np.nonzero(result))

def draw_bounding_box(im, box):
    if not isinstance(box, list):
        box = [box]

    im = np.array(im, dtype=np.uint8)

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)

    # Create a Rectangle patch
    for c_box in box:
        rect = patches.Rectangle((c_box[1], c_box[0]), c_box[3] - c_box[1], c_box[2] - c_box[0],
                                  linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()
