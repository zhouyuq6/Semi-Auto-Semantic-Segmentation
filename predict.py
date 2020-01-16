from u_net import *
from load_data import *
from preprocessing import *
from MeDIT.Visualization import Imshow3DArray
import matplotlib.pyplot as plt


def show_roi(image_og, ROI=None):
    """
    Use Imshow3D to display both image and ROI
    :return: N/A
    """
    if isinstance(ROI, list):
        my_roi = []
        for roi in ROI:
            image_mask = (np.moveaxis(roi[..., 0], 0, -1)).astype(np.uint8)
            my_roi += [image_mask]
    elif isinstance(ROI, type(image_og)):
        my_roi = (np.moveaxis(ROI[..., 0], 0, -1)).astype(np.uint8)
    else:
        return False

    image = np.moveaxis(image_og[..., 0], 0, -1)
    image = normalize_mm(image)
    Imshow3DArray(image, ROI=my_roi)
    return


def predict_by_slice(test_id, model, data_path, weights):
    case_num = len(test_id)
    A_loss = 0
    A_dsc = 0
    for cid in test_id:
        # LOAD DATA
        case_name = "case_{:05d}".format(cid)
        print(case_name)
        case_path = data_path + case_name

        # DATA PRE-PROCESSING
        [bot_img, top_roi], bot_roi = test_data(case_path)

        # LOAD MODEL
        weight_path = weights
        model = model
        model.load_weights(weight_path)

        # PREDICT A: Feeding the network with the same kind of data as training data
        # Need to change to the same input pre-processing procedure as that in data_generator
        inputs = np.asarray([bot_img, top_roi])[:, :, :, :, 0]
        inputs_t = inputs.transpose(1, 2, 3, 0)
        pred_roi = model.predict(inputs_t, verbose=1)
        pred_roi = binary_converter(pred_roi, 0.5)

        A_loss = A_loss + dice_loss_np(pred_roi, bot_roi)
        A_dsc = A_dsc + dice_coeff_np(pred_roi, bot_roi)

        # VISUALIZATION
        # Need: from MeDIT.Visualization import Imshow3DArray
        # show_roi(bot_img, ROI=[pred_roi])

    A_loss /= case_num
    A_dsc /= case_num
    return A_loss, A_dsc


def predict_from_roi(test_id, model, data_path, weights):
    case_num = len(test_id)
    B_loss = 0
    B_dsc = 0
    for cid in test_id:
        # LOAD DATA
        case_name = "case_{:05d}".format(cid)
        print(case_name)
        case_path = data_path + case_name

        # DATA PRE-PROCESSING
        [bot_img, top_roi], bot_roi = test_data(case_path)

        # LOAD MODEL
        weight_path = weights
        model = model
        model.load_weights(weight_path)

        # PREDICT B: Feeding the network its ground truth neighbour ROI and image each layer
        max_id = find_max_slice(bot_roi)
        pred_roi = np.ndarray(shape=bot_img.shape, dtype=np.float32)
        pred_roi[max_id] = bot_roi[max_id]

        for sid in reversed(range(0, max_id)):
            input_roi = bot_roi[sid - 1]
            input_img = bot_img[sid]
            inputs = np.asarray([input_img, input_roi])
            inputs_t = inputs.transpose(3, 1, 2, 0)
            pred_roi[sid] = binary_converter(model.predict(inputs_t, verbose=1), 0.5)

        for sid in range(max_id + 1, bot_img.shape[0]):
            input_roi = bot_roi[sid - 1]
            input_img = bot_img[sid]
            inputs = np.asarray([input_img, input_roi])
            inputs_t = inputs.transpose(3, 1, 2, 0)
            pred_roi[sid] = binary_converter(model.predict(inputs_t, verbose=1), 0.5)

        B_loss = B_loss + dice_loss_np(bot_roi, pred_roi)
        B_dsc = B_dsc + dice_coeff_np(bot_roi, pred_roi)

        # VISUALIZATION
        # Need: from MeDIT.Visualization import Imshow3DArray
        # show_roi(bot_img, ROI=[pred_roi])

    B_loss /= case_num
    B_dsc /= case_num
    return B_loss, B_dsc


def predict_from_self(test_id, model, data_path, weights):
    case_num = len(test_id)
    C_loss = 0
    C_dsc = 0
    for cid in test_id:
        # LOAD DATA
        case_name = "case_{:05d}".format(cid)
        print(case_name)
        case_path = data_path + case_name

        # DATA PRE-PROCESSING
        [bot_img, top_roi], bot_roi = test_data(case_path)

        # LOAD MODEL
        weight_path = weights
        model = model
        model.load_weights(weight_path)

        # PREDICT C: Feeding the network its predicted neighbour ROI and image each layer
        max_id = find_max_slice(bot_roi)
        pred_roi = np.ndarray(shape=bot_img.shape, dtype=np.float32)
        pred_roi[max_id] = bot_roi[max_id]

        for sid in reversed(range(0, max_id)):
            input_roi = pred_roi[sid - 1]
            input_img = bot_img[sid]
            inputs = np.asarray([input_img, input_roi])
            inputs_t = inputs.transpose(3, 1, 2, 0)
            pred_roi[sid] = binary_converter(model.predict(inputs_t, verbose=1), 0.5)

        for sid in range(max_id + 1, bot_img.shape[0]):
            input_roi = pred_roi[sid - 1]
            input_img = bot_img[sid]
            inputs = np.asarray([input_img, input_roi])
            inputs_t = inputs.transpose(3, 1, 2, 0)
            pred_roi[sid] = binary_converter(model.predict(inputs_t, verbose=1), 0.5)

        C_loss = C_loss + dice_loss_np(pred_roi, bot_roi)
        C_dsc = C_dsc + dice_coeff_np(pred_roi, bot_roi)

        # VISUALIZATION
        # Need: from MeDIT.Visualization import Imshow3DArray
        # show_roi(bot_img, ROI=[pred_roi])

    C_loss /= case_num
    C_dsc /= case_num
    return C_loss, C_dsc
