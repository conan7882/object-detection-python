
import numpy as np

# from features import Features
from selective_search import SelectiveSearch
import utils
from factory import draw_bounding_box

IMPATH = '../data/p.png'
LABLEPATH = '../data/seg_p.mat'


if __name__ == '__main__':
    im = utils.load_image(IMPATH, read_channel=3)
    im = np.squeeze(im, axis=0)
    print(im.shape)

    label = utils.load_image_from_mat(LABLEPATH, 'gray_label', 'int64')
    n_region = np.amax(label)+1
    print(label.shape)
    print(np.amax(label))

    color_space = ['hsv', 'lab', 'rgb']
    mode = [('C', 'S', 'F', 'T'), ('C', 'S', 'F')]
    ss = SelectiveSearch(im, label, color_space, mode)
    box = ss.hierarchical_segmentation()
    # box = sss.ranked_box()
    # # print(len(box))
    box_draw = list([box[i][1] for i in range(min(50, len(box)))])
    # # print(box_draw)
    draw_bounding_box(im, box_draw)


