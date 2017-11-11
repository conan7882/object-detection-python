from scipy import misc
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

def load_image(im_path, read_channel=None, resize=None):
    # im = cv2.imread(im_path, self._cv_read)
    if read_channel is None:
        im = misc.imread(im_path)
    elif read_channel == 3:
        im = misc.imread(im_path, mode='RGB')
    else:
        im = misc.imread(im_path, flatten=True)

    if len(im.shape) < 3:
        try:
            im = misc.imresize(im, (resize[0], resize[1], 1))
        except TypeError:
            pass
        im = np.reshape(im, [1, im.shape[0], im.shape[1], 1])
    else:
        try:
            im = misc.imresize(im, (resize[0], resize[1], im.shape[2]))
        except TypeError:
            pass
        im = np.reshape(im, [1, im.shape[0], im.shape[1], im.shape[2]])
    return im

def intensity_to_rgb(intensity, cmap='jet', normalize=False):
    """
    This function is copied from `tensorpack
    <https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/utils/viz.py>`__.
    Convert a 1-channel matrix of intensities to an RGB image employing
    a colormap.
    This function requires matplotlib. See `matplotlib colormaps
    <http://matplotlib.org/examples/color/colormaps_reference.html>`_ for a
    list of available colormap.

    Args:
        intensity (np.ndarray): array of intensities such as saliency.
        cmap (str): name of the colormap to use.
        normalize (bool): if True, will normalize the intensity so that it has
            minimum 0 and maximum 1.

    Returns:
        np.ndarray: an RGB float32 image in range [0, 255], a colored heatmap.
    """

    # assert intensity.ndim == 2, intensity.shape
    intensity = intensity.astype("float")

    if normalize:
        intensity -= intensity.min()
        intensity /= intensity.max()

    if intensity.ndim == 3:
        return intensity.astype('float32') * 255.0

    cmap = plt.get_cmap(cmap)
    intensity = cmap(intensity)[..., :3]
    return intensity.astype('float32') * 255.0


def load_image_from_mat(file_path, name, datatype):
    matfile = loadmat(file_path)
    mat = matfile[name].astype(datatype)
    return mat


def assert_type(v, tp):
    """
    Assert type of input v be type tp
    """
    assert isinstance(v, tp),\
        "Expect " + str(tp) + ", but " + str(v.__class__) + " is given!"