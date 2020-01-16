from pathlib import Path
import nibabel as nib
from preprocessing import *
import random
import os
from MeDIT.DataAugmentor import DataAugmentor2D, AugmentParametersGenerator


def load_case(case_id, data_path):
    """
    Load image and roi data from .nii files
    :param case_id:
    :return: 3D array with shape [slice x row x col]
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise OSError("Data Path {} could not be found".format(str(data_path)))

    try:
        case_id_name = "case_{:05d}".format(case_id)
    except ValueError:
        case_id_name = case_id

    case_path = data_path / case_id_name
    if not case_path.exists():
        raise ValueError("Case {} could not be found".format(case_id_name))

    vol = nib.load(str(case_path / "imaging.nii.gz"))
    seg = nib.load(str(case_path / "segmentation.nii.gz"))
    return vol, seg


def load_nii(case_path):
    """
    Load image and roi data from .nii files
    :param case_id:
    :return: 3D array with shape [slice x row x col]
    """
    data_path = Path(case_path)
    if not data_path.exists():
        raise OSError("Data Path {} could not be found".format(str(data_path)))

    vol = nib.load(str(data_path / "imaging.nii.gz"))
    seg = nib.load(str(data_path / "segmentation.nii.gz"))
    return vol, seg


def get_data_by_case(data_path):
    """
    Get image and roi data directly from a file by its case_id. This is for getting testing data
    :param case_id:
    :return: top_test: upper layer roi
             bot_test: lower layer image
             mask_test: lower layer roi
    """
    vol, seg = load_nii(data_path)
    vol_data = vol.get_data()
    seg_all = seg.get_data()
    seg_data = np.zeros((seg_all.shape[0], seg_all.shape[1], seg_all.shape[2]), dtype=np.float32)
    seg_data[np.greater_equal(seg_all, 1)] = 1
    top_test = seg_data[:-1, ...]
    bot_test = vol_data[1:, ...]
    mask_test = seg_data[1:, ...]
    return top_test, bot_test, mask_test


def load_data_from_case(case_id, data_path, target_path):
    """
    Load CT scans from data path by case id, pack (image, roi and neighbour roi) of each scan
    as an individual npy file,  and save to target_path
    :param case_id:
    :param data_path:
    :param target_path:
    :return:
    """
    case_path = Path(target_path)
    if not case_path.exists():
        case_path.mkdir()
    # Get vol, seg data from images
    vol, seg = load_case(case_id, data_path)
    vol_data = vol.get_data()
    seg_all = seg.get_data()
    # Extract kidney mask labels only
    seg_data = np.zeros((seg_all.shape[0], seg_all.shape[1], seg_all.shape[2]), dtype=np.uint8)
    seg_data[np.greater_equal(seg_all, 1)]=1

    empty = []
    non_empty = []
    for slice in range(seg_data.shape[0]):
        if np.sum(seg_data[slice]) == 0:
            empty = empty + [slice]
        else:
            non_empty = non_empty + [slice]

    random.shuffle(empty)
    empty_len = int(len(empty) / 2)
    slice_ids = non_empty + empty[:empty_len]
    slice_ids.sort()
    for sid in slice_ids:
        offset = random.choice([-5,-4,-3,-2,-1,1,2,3,4,5])
        roi_id = sid + offset
        if roi_id < 0 and not sid == 0:
            roi_id = 0
        elif roi_id < 0 and sid == 0:
            roi_id = sid + random.choice([1, 2, 3, 4, 5])
        elif roi_id > seg_data.shape[0] - 1 and not sid == seg_data.shape[0] - 1:
            roi_id = seg_data.shape[0] - 1
        elif roi_id > seg_data.shape[0] - 1 and sid == seg_data.shape[0] - 1:
            roi_id = sid + random.choice([-5,-4,-3,-2,-1])
        else:
            pass
        if sid == roi_id:
            print(sid, roi_id)
        tup = np.ndarray((3, RAW_WIDTH, RAW_HEIGHT), dtype=np.float32)
        tup[0] = vol_data[sid]
        tup[1] = vol_data[roi_id]
        tup[2] = seg_data[roi_id]
        img_path = (target_path + '/case_{:03}_s{}.npy').format(case_id, sid)
        np.save(img_path, tup)
        roi = seg_data[sid]
        roi_path = (target_path + '/case_{:03}_s{}_roi.npy').format(case_id, sid)
        np.save(roi_path, roi)
    return True


def load_data_from_case_alt(case_id, data_path, target_path):
    """
    Same as load data from case except that neighbour roi is chosen from +- 5 layers away
    (load data from case is chosen randomly within -5 ~ +5 layers
    :param case_id:
    :param data_path:
    :param target_path:
    :return:
    """
    case_path = Path(target_path)
    if not case_path.exists():
        case_path.mkdir()
    # Get vol, seg data from images
    vol, seg = load_case(case_id, data_path)
    vol_data = vol.get_data()
    seg_all = seg.get_data()
    # Extract kidney mask labels only
    seg_data = np.zeros((seg_all.shape[0], seg_all.shape[1], seg_all.shape[2]), dtype=np.uint8)
    seg_data[np.greater_equal(seg_all, 1)]=1
    for sid in range(vol_data.shape[0]):
        if sid < 5:
            roi_id = 5
        elif sid > vol_data.shape[0] - 6:
            roi_id = -5
        else:
            roi_id = random.choice([-5,5])
        tup = np.ndarray((3, RAW_WIDTH, RAW_HEIGHT), dtype=np.float32)
        tup[0] = vol_data[sid]
        tup[1] = vol_data[sid + roi_id]
        tup[2] = seg_data[sid + roi_id]
        img_path = (target_path + '/case_{:03}_s{}.npy').format(case_id, sid)
        np.save(img_path, tup)

        roi = seg_data[sid]
        roi_path = (target_path + '/case_{:03}_s{}_roi.npy').format(case_id, sid)
        np.save(roi_path, roi)
    return True


def data_generator(data_path, batch_size):
    """
    training data generator. Including data augmentation, croping, resizing, and normalization
    :param data_path:
    :param batch_size:
    :return:
    """
    while True:
        data_path = Path(data_path)
        if not data_path.exists():
            raise OSError("Data Path {} could not be found".format(str(data_path)))
        bot_img = np.ndarray((batch_size, IMAGE_WIDTH, IMAGE_HEIGHT), dtype=np.float32)
        top_roi = np.ndarray((batch_size, IMAGE_WIDTH, IMAGE_HEIGHT), dtype=np.float32)
        bot_roi = np.ndarray((batch_size, IMAGE_WIDTH, IMAGE_HEIGHT), dtype=np.float32)
        # Data Augmentation
        augmenter = DataAugmentor2D()
        parameter_generator = AugmentParametersGenerator()
        for case in range(batch_size):
            file_name = random.choice(os.listdir(str(data_path)))
            if file_name.endswith("_roi.npy"):
                gt_file = file_name
                case_file = file_name[:-8] + ".npy"

            else:
                case_file = file_name
                gt_file = case_file[:-4] + "_roi.npy"

            # Data Augmentation
            aug_parameter_dict = {'stretch_x': 0.1,
                                  'stretch_y': 0.1,
                                  'shear': 0.1,
                                  'rotate_z_angle': 5,
                                  'horizontal_flip': True}
            parameter_generator.RandomParameters(aug_parameter_dict)
            aug_param = parameter_generator.GetRandomParametersDict()
            # Get Input Data
            tup = np.load(data_path / case_file)
            # Get Input Image
            img = np.squeeze(tup[0])
            aug_img = augmenter.Execute(img, aug_parameter=aug_param)
            norm_img = normalize_mm(aug_img[32:RAW_WIDTH-32, 32:RAW_HEIGHT-32])
            bot_img[case] = resize(norm_img, (IMAGE_WIDTH, IMAGE_HEIGHT), preserve_range=True)
            # Get Input ROI
            roi = np.squeeze(tup[2])
            aug_roi = augmenter.Execute(roi, aug_parameter=aug_param)
            crop_roi = aug_roi[32:RAW_WIDTH-32, 32:RAW_HEIGHT-32]
            resize_roi = resize(crop_roi, (IMAGE_WIDTH, IMAGE_HEIGHT), preserve_range=True)
            top_roi[case] = binary_converter(resize_roi, 0.5)
            # Get Ground Truth ROI
            gt = np.load(data_path / gt_file)
            gt = np.squeeze(gt)
            aug_gt = augmenter.Execute(gt, aug_parameter=aug_param)
            crop_gt = aug_gt[32:RAW_WIDTH-32, 32:RAW_HEIGHT-32]
            resize_gt = resize(crop_gt, (IMAGE_WIDTH, IMAGE_HEIGHT), preserve_range=True)
            bot_roi[case] = binary_converter(resize_gt, 0.5)

        inputs = np.asarray([bot_img, top_roi])
        inputs_t = inputs.transpose(1,2,3,0)
        # bot_img = bot_img[..., np.newaxis]
        # top_roi = top_roi[..., np.newaxis]
        bot_roi = bot_roi[..., np.newaxis]
        yield inputs_t, bot_roi


def test_generator(data_path, batch_size):
    """
    Validation data generator. No augmentation.
    :param data_path:
    :param batch_size:
    :return:
    """
    while True:
        data_path = Path(data_path)
        if not data_path.exists():
            raise OSError("Data Path {} could not be found".format(str(data_path)))
        bot_img = np.ndarray((batch_size, IMAGE_WIDTH, IMAGE_HEIGHT), dtype=np.float32)
        top_roi = np.ndarray((batch_size, IMAGE_WIDTH, IMAGE_HEIGHT), dtype=np.float32)
        bot_roi = np.ndarray((batch_size, IMAGE_WIDTH, IMAGE_HEIGHT), dtype=np.float32)

        for case in range(batch_size):
            file_name = random.choice(os.listdir(str(data_path)))
            if file_name.endswith("_roi.npy"):
                gt_file = file_name
                case_file = file_name[:-8] + ".npy"

            else:
                case_file = file_name
                gt_file = case_file[:-4] + "_roi.npy"

            # Get Input Data
            tup = np.load(data_path / case_file)
            # Get Input Image
            img = np.squeeze(tup[0])
            norm_img = normalize_mm(img[32:RAW_WIDTH-32, 32:RAW_HEIGHT-32])
            bot_img[case] = resize(norm_img, (IMAGE_WIDTH, IMAGE_HEIGHT), preserve_range=True)
            # Get Input ROI
            roi = np.squeeze(tup[2])
            crop_roi = roi[32:RAW_WIDTH-32, 32:RAW_HEIGHT-32]
            resize_roi = resize(crop_roi, (IMAGE_WIDTH, IMAGE_HEIGHT), preserve_range=True)
            top_roi[case] = binary_converter(resize_roi, 0.5)
            # Get Ground Truth ROI
            gt = np.load(data_path / gt_file)
            gt = np.squeeze(gt)
            crop_gt = gt[32:RAW_WIDTH-32, 32:RAW_HEIGHT-32]
            resize_gt = resize(crop_gt, (IMAGE_WIDTH, IMAGE_HEIGHT), preserve_range=True)
            bot_roi[case] = binary_converter(resize_gt, 0.5)

        inputs = np.asarray([bot_img, top_roi])
        inputs_t = inputs.transpose(1,2,3,0)
        # bot_img = bot_img[..., np.newaxis]
        # top_roi = top_roi[..., np.newaxis]
        bot_roi = bot_roi[..., np.newaxis]
        yield inputs_t, bot_roi


def test_data(data_path):
    """
    Generate testing data
    :param data_path:
    :return:
    """
    vol, seg = load_nii(data_path)
    vol_data = vol.get_data()
    seg_all = seg.get_data()
    seg_data = np.zeros((seg_all.shape[0], seg_all.shape[1], seg_all.shape[2]), dtype=np.uint8)
    seg_data[np.greater_equal(seg_all, 1)] = 1
    batch_size = seg_data.shape[0]
    slice_ids = [i for i in range(batch_size)]
    bot_img = np.ndarray((batch_size, IMAGE_WIDTH, IMAGE_HEIGHT), dtype=np.float32)
    top_roi = np.ndarray((batch_size, IMAGE_WIDTH, IMAGE_HEIGHT), dtype=np.float32)
    bot_roi = np.ndarray((batch_size, IMAGE_WIDTH, IMAGE_HEIGHT), dtype=np.float32)

    # Random
    # for sid in slice_ids:
    #     offset = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
    #     roi_id = sid + offset
    #     if roi_id < 0 and not sid == 0:
    #         roi_id = 0
    #     elif roi_id < 0 and sid == 0:
    #         roi_id = sid + random.choice([1, 2, 3, 4, 5])
    #     elif roi_id > seg_data.shape[0] - 1 and not sid == seg_data.shape[0] - 1:
    #         roi_id = seg_data.shape[0] - 1
    #     elif roi_id > seg_data.shape[0] - 1 and sid == seg_data.shape[0] - 1:
    #         roi_id = sid + random.choice([-5, -4, -3, -2, -1])
    #     else:
    #         pass
    #     if sid == roi_id:
    #         print(sid, roi_id)

    # Fixed
    dist = 5
    for sid in slice_ids:
        if sid < dist:
            roi_id = sid + dist
        elif sid > vol_data.shape[0] - dist - 1:
            roi_id = sid - dist
        else:
            roi_id = sid + random.choice([-dist, dist])

        # Get Input Image
        img = np.squeeze(vol_data[sid])
        norm_img = normalize_mm(img[32:RAW_WIDTH-32, 32:RAW_HEIGHT-32])
        bot_img[sid] = resize(norm_img, (IMAGE_WIDTH, IMAGE_HEIGHT), preserve_range=True)
        # Get Input ROI
        roi = np.squeeze(seg_data[roi_id])
        crop_roi = roi[32:RAW_WIDTH-32, 32:RAW_HEIGHT-32]
        resize_roi = resize(crop_roi, (IMAGE_WIDTH, IMAGE_HEIGHT), preserve_range=True)
        top_roi[sid] = binary_converter(resize_roi, 0.5)
        # Get Ground Truth ROI
        gt = np.squeeze(seg_data[sid])
        crop_gt = gt[32:RAW_WIDTH-32, 32:RAW_HEIGHT-32]
        resize_gt = resize(crop_gt, (IMAGE_WIDTH, IMAGE_HEIGHT), preserve_range=True)
        bot_roi[sid] = binary_converter(resize_gt, 0.5)

    bot_img = bot_img[..., np.newaxis]
    top_roi = top_roi[..., np.newaxis]
    bot_roi = bot_roi[..., np.newaxis]
    return [bot_img, top_roi], bot_roi
