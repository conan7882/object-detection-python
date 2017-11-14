
import numpy as np
import cv2
import matplotlib.pyplot as plt

from factory import find_adjacent_region, find_all_adjacent_region
from features import Features
from utils import intensity_to_rgb

class SelectiveSearch(object):
    def __init__(self, rgb_im, label_im, color_space, mode):
        if not isinstance(mode, list):
            mode = [mode]
        for mode_i in mode:
            for mode_s in mode_i:
                print(mode_s)
                assert mode_s in ['C', 'T', 'S', 'F']

        if not isinstance(color_space, list):
            color_space = [color_space.lower()]
        for c_i in color_space:
            print(c_i)
            assert c_i in ['hsv', 'lab', 'rgb']

        self._label = label_im
        self._n_region = np.amax(label_im) + 1
        self._rgb = rgb_im
        self._mode = mode
        self._c_space = color_space

        # CV_RGB2HSV CV_BGR2Lab
        # cv2.cvtColor(src, 'CV_RGB2HSV') 
    def hierarchical_segmentation(self):
        color_code_dict = {'hsv': cv2.COLOR_BGR2HSV,
                           'lab': cv2.COLOR_BGR2Lab,
                           'rgb': None}
        ss_group = []
        for color in self._c_space:
            print(color_code_dict[color])
            im_c = cv2.cvtColor(self._rgb, color_code_dict[color]) if not color_code_dict[color] is None else self._rgb
            for mode in self._mode:
                ss_group.append(SelectiveSearchSingle(im_c, self._label, 
                                                      self._n_region, mode=mode))

        re_box = []
        for sss in ss_group:
            re_box.extend(sss.ranked_box())

        def getKey(item):
            return item[0]

        return sorted(re_box, key=getKey)

class SelectiveSearchSingle(object):
    """docstring for SelectiveSearch"""
    def __init__(self, image, label_im, n_region, mode=('C', 'T', 'S', 'F')):
        self._label_im = label_im
        self._im = image
        
        self._feats = Features(image, label_im, n_region, mode=mode)
        self._n_region = n_region

        all_pair = find_all_adjacent_region(label_im)
        self._adj_dict = {}
        for pair in all_pair:
            if pair[0] < pair[1]:
                self._adj_dict[(pair[0], pair[1])] = self._feats.similarity(pair[0], pair[1])

        # print(max(self._adj_dict, key=self._adj_dict.get))
        # self._merge_region(122, 271, n_region)
    def ranked_box(self):
        self._g_box = self.hierarchical_grouping()
        return self._gen_rank()

    def _gen_rank(self):
        num_box = len(self._g_box)
        num_init = self._n_region
        rank = (list([(np.random.rand() * min(num_box - label, num_box - num_init + 1), 
                       self._g_box[label])
                for label in range(num_box)]))

        return rank

    def hierarchical_grouping(self):
        n_region = self._n_region 
        new_id = n_region
        self._merge_order = []
        while (n_region > 1):
            n_region -= 1
            max_similar_pair = max(self._adj_dict, key=self._adj_dict.get)
            self._merge_region(max_similar_pair[0], max_similar_pair[1], new_id)
            self._merge_order += (max_similar_pair[0], max_similar_pair[1], new_id),
            new_id += 1

        # return self._feats.bounding_box
        return list([self._feats.bounding_box[i] for i in range(new_id)])

    def _merge_region(self, i, j, new_id):
        i, j = min(i, j), max(i, j)
        del self._adj_dict[(i, j)]
        self._feats.merge_feature(i, j, new_id)

        nei_i = [(k, l) for (k, l) in self._adj_dict if k == i or l == i]
        for nei in nei_i:
            del self._adj_dict[nei]
            nei_id = nei[0] if i == nei[1] else nei[1]
            self._adj_dict[(nei_id, new_id)] = self._feats.similarity(nei_id, new_id)

        nei_j = [(k, l) for (k, l) in self._adj_dict if k == j or l == j]
        for nei in nei_j:
            del self._adj_dict[nei]
            nei_id = nei[0] if j == nei[1] else nei[1]
            self._adj_dict[(nei_id, new_id)] = self._feats.similarity(nei_id, new_id) 

    def show_merge_process(self):
        label_im = self._label_im

        img = plt.imshow(0.5 * intensity_to_rgb(label_im, normalize=True))
        for merge_event in self._merge_order:

            label_im[label_im==merge_event[0]] = merge_event[2]
            label_im[label_im==merge_event[1]] = merge_event[2]

            img.set_data(0.5 * intensity_to_rgb(label_im, normalize=True))
            plt.pause(.1)
            plt.draw()




        


        




    
        
        