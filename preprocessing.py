from skimage.transform import resize, rescale
from random import shuffle
from u_net import *


def normalize_mm(imgs):
    """
    Normalization using max and min
    :param imgs: shape = [slice x row x col]
    :return:
    """
    imgs = imgs - np.min(imgs)
    imgs = imgs / np.max(imgs)
    return imgs


def binary_converter(imgs, threshold):
    """
    Convert image to binary image, based on threshold
    :param imgs: original image array
    :param threshold: float, 0~1
    :return: Binary image array
    """
    binary_img = np.zeros((imgs.shape), dtype=imgs.dtype)
    binary_img[np.greater_equal(imgs, threshold)] = 1
    binary_img[np.less(imgs, threshold)] = 0
    return binary_img


def index_split(samples, str_case, test_ratio, val_ratio):
    idx = [i+str_case for i in range(samples)]
    shuffle(idx)
    b1 = int(samples*(1-val_ratio-test_ratio))
    b2 = int(samples*(1-test_ratio))
    train_case_id = idx[:b1]
    train_case_id.sort()
    val_case_id = idx[b1:b2]
    val_case_id.sort()
    test_case_id = idx[b2:]
    test_case_id.sort()
    return train_case_id, val_case_id, test_case_id


def find_max_slice(roi):
    max_slice = 0
    max_id = 0
    for sid in range(roi.shape[0]):
        slice_sum = np.sum(roi[sid])
        if slice_sum > max_slice:
            max_slice = slice_sum
            max_id = sid
    return max_id