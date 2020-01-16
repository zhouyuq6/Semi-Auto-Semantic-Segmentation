from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Flatten, UpSampling2D, Dropout, BatchNormalization, Activation
from keras import optimizers
from keras.regularizers import l2, l1
from keras import backend as K
import numpy as np

# MACROS
samples = 150
str_case = 0
RAW_HEIGHT = 512
RAW_WIDTH = 512
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
N_CH = 1
IMAGE_ORDERING = 'channels_last'
MERGE_AXIS = -1
SMOOTH = 1e-5


# LOSS & METRIC FUNCTIONS
def dice_coeff_np(y_true, y_pred):
    """
    Dice coefficient function using numpy.
    Tensorflow/Keras backend calculates tensors, hard to estimate metrics of testing data (in numpy array).
    :param y_true: ground truth roi
    :param y_pred: predicted roi
    :return: dice coefficient, float between 0 and 1
    """
    y_true_f = np.ravel(y_true)
    y_pred_f = np.ravel(y_pred)
    intersection = np.sum((y_true_f * y_pred_f))
    union = (np.sum(y_true_f * y_true_f)) + (np.sum(y_pred_f * y_pred_f))
    return (2. * intersection) / (union + SMOOTH)


def dice_loss_np(y_true, y_pred):
    """
    Soft Dice loss function using numpy.
    :param y_true: ground truth roi
    :param y_pred: predicted roi
    :return: dice loss, float between 0 and 1
    """
    return 1. - dice_coeff_np(y_true, y_pred)


def dice_coeff(y_true, y_pred):
    """
    Dice coefficient function using Keras/Tensorflow backend.
    Reference: https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py#L23
    :param y_true: ground truth roi
    :param y_pred: predicted roi
    :return: dice coefficient, float between 0 and 1
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum((y_true_f * y_pred_f))
    union = (K.sum(y_true_f * y_true_f)) + (K.sum(y_pred_f * y_pred_f))
    return (2. * intersection) / (union + SMOOTH)


def dice_loss(y_true, y_pred):
    """
    Soft Dice loss function using Keras/Tensorflow backend.
    :param y_true: ground truth roi
    :param y_pred: predicted roi
    :return: dice loss, float between 0 and 1
    """
    return 1. - dice_coeff(y_true, y_pred)


def jaccard(y_true, y_pred):
    """
    Jaccard function or IoU.
    :param y_true: ground truth roi
    :param y_pred: predicted roi
    :return: Jaccard coefficient, float between 0 and 1
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum((y_true_f * y_pred_f))
    union = (K.sum(y_true_f * y_true_f)) + (K.sum(y_pred_f * y_pred_f))
    return (intersection) / (union - intersection + SMOOTH)


def jaccard_loss(y_true, y_pred):
    """
    Jaccard loss function, 1 - Jaccard coeff
    :param y_true: ground truth roi
    :param y_pred: predicted roi
    :return: Jaccard loss, float between 0 and 1
    """
    return 1 - jaccard(y_true,y_pred)


def u_net():
    """
    Vanilla U-Net Structure
    :return: A u-net model
    """
    inputs = Input((IMAGE_HEIGHT, IMAGE_WIDTH, N_CH))

    #Encoding
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    #Bottom Layer
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(pool4)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv5)
    # Decoding
    up1 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=MERGE_AXIS)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(up1)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv6)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=MERGE_AXIS)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(up2)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv7)

    up3 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=MERGE_AXIS)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(up3)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv8)

    up4 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=MERGE_AXIS)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(up4)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv9)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer=optimizers.adam(lr=1e-3, decay=0.1), loss=jaccard_loss, metrics=[jaccard])

    return model


def u_net_deep():
    """
    Deeper U-Net comparing to u_net()
    :return:
    """
    inputs = Input((IMAGE_HEIGHT, IMAGE_WIDTH, N_CH))

    #Encoding
    # BatchNormalization
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)

    conve = Conv2D(256, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(pool4)
    conve = Dropout(0.2)(conve)
    conve = Conv2D(256, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conve)
    poole = MaxPooling2D((2, 2))(conve)

    #Bottom Layer
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(poole)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv5)

    # Decoding
    upd = concatenate([UpSampling2D(size=(2, 2))(conv5), conve], axis=MERGE_AXIS)
    convd = Conv2D(256, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(upd)
    convd = Dropout(0.2)(convd)
    convd = Conv2D(256, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(convd)

    up1 = concatenate([UpSampling2D(size=(2, 2))(convd), conv4], axis=MERGE_AXIS)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(up1)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv6)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=MERGE_AXIS)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(up2)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv7)

    up3 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=MERGE_AXIS)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(up3)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv8)

    up4 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=MERGE_AXIS)
    conv9 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(up4)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv9)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer=optimizers.adam(lr=1e-3, decay=0.1), loss = dice_loss, metrics = [dice_coeff])
    return model


def u_net_multi():
    """
    A U-Net structure takes two inputs. Both inputs experienced convolution and pooling.
    :return:
    """
    input1 = Input((IMAGE_HEIGHT, IMAGE_WIDTH, N_CH))
    input2 = Input((IMAGE_HEIGHT, IMAGE_WIDTH, N_CH))

    conv01 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(input1)
    conv01 = Dropout(0.2)(conv01)
    conv01 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv01)
    pool01 = MaxPooling2D((2, 2))(conv01)

    conv02 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(input2)
    conv02 = Dropout(0.2)(conv02)
    conv02 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv02)
    pool02 = MaxPooling2D((2, 2))(conv02)

    inputs = concatenate([pool01, pool02], axis=MERGE_AXIS)

    #Encoding
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    #Bottom Layer
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(pool4)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv5)
    # Decoding
    up1 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=MERGE_AXIS)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(up1)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv6)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=MERGE_AXIS)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(up2)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv7)

    up3 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=MERGE_AXIS)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(up3)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv8)

    up4 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=MERGE_AXIS)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(up4)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv9)

    up5 = concatenate([UpSampling2D(size=(2, 2))(conv9), conv01, conv02], axis=MERGE_AXIS)
    conv10 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(up5)
    conv10 = Dropout(0.2)(conv10)
    conv10 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv10)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv10)

    model = Model(inputs=[input1, input2], outputs=[outputs])

    model.compile(optimizer=optimizers.adam(lr=1e-3, decay=0.01), loss=dice_loss, metrics=[dice_coeff])

    return model


def u_net_multi_channel():
    """
    A U-Net structure takes a two channel inputs. Use u_net_bn(input_size=(IMAGE_HEIGHT, IMAGE_WIDTH, 2)) instead.
    :return:
    """
    inputs = Input((IMAGE_HEIGHT, IMAGE_WIDTH, 2))
    #Encoding
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)

    conve = Conv2D(256, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(pool4)
    conve = Dropout(0.2)(conve)
    conve = Conv2D(256, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conve)
    poole = MaxPooling2D((2, 2))(conve)

    #Bottom Layer
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(poole)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv5)

    # Decoding
    upd = concatenate([UpSampling2D(size=(2, 2))(conv5), conve], axis=MERGE_AXIS)
    convd = Conv2D(256, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(upd)
    convd = Dropout(0.2)(convd)
    convd = Conv2D(256, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(convd)

    up1 = concatenate([UpSampling2D(size=(2, 2))(convd), conv4], axis=MERGE_AXIS)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(up1)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv6)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=MERGE_AXIS)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(up2)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv7)

    up3 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=MERGE_AXIS)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(up3)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv8)

    up4 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=MERGE_AXIS)
    conv9 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(up4)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format=IMAGE_ORDERING)(conv9)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer=optimizers.adam(lr=1e-3, decay=0.1), loss = dice_loss, metrics = [dice_coeff])
    return model


def conv_bn_relu(**kwargs):
    """
    Convolution - batch nomalization - ReLU module
    :param kwargs:
    :return:
    """
    filters = kwargs['filters']
    kernel_size = kwargs['kernel_size']
    strides = kwargs.setdefault('strides', (1, 1))
    padding = kwargs.setdefault('padding', 'same')
    kernel_initializer = kwargs.setdefault('kernal_initializer', 'he_normal')
    data_format = kwargs.setdefault('data_format', 'channels_last')

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      data_format=data_format)(input)
        norm = BatchNormalization(axis=-1)(conv)
        actv = Activation('relu')(norm)
        return actv

    return f


def u_net_bn(input_size):
    """
    A general U-Net structure that takes different input sizes.
    :param input_size: A tuple of input size with 'channel-last' format e.g. (width, height, channel)
    :return:
    """
    inputs = Input(input_size)
    #Encoding
    conv1 = conv_bn_relu(filters=16, kernel_size=(3, 3))(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = conv_bn_relu(filters=16, kernel_size=(3, 3))(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = conv_bn_relu(filters=32, kernel_size=(3, 3))(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = conv_bn_relu(filters=32, kernel_size=(3, 3))(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = conv_bn_relu(filters=64, kernel_size=(3, 3))(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = conv_bn_relu(filters=64, kernel_size=(3, 3))(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = conv_bn_relu(filters=128, kernel_size=(3, 3))(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = conv_bn_relu(filters=128, kernel_size=(3, 3))(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)

    conve = conv_bn_relu(filters=256, kernel_size=(3, 3))(pool4)
    conve = Dropout(0.2)(conve)
    conve = conv_bn_relu(filters=256, kernel_size=(3, 3))(conve)
    poole = MaxPooling2D((2, 2))(conve)

    #Bottom Layer
    conv5 = conv_bn_relu(filters=512, kernel_size=(3, 3))(poole)
    conv5 = Dropout(0.2)(conv5)
    conv5 = conv_bn_relu(filters=512, kernel_size=(3, 3))(conv5)

    # Decoding
    upd = concatenate([UpSampling2D(size=(2, 2))(conv5), conve], axis=MERGE_AXIS)
    convd = conv_bn_relu(filters=256, kernel_size=(3, 3))(upd)
    convd = Dropout(0.2)(convd)
    convd = conv_bn_relu(filters=256, kernel_size=(3, 3))(convd)

    up1 = concatenate([UpSampling2D(size=(2, 2))(convd), conv4], axis=MERGE_AXIS)
    conv6 = conv_bn_relu(filters=128, kernel_size=(3, 3))(up1)
    conv6 = Dropout(0.2)(conv6)
    conv6 = conv_bn_relu(filters=128, kernel_size=(3, 3))(conv6)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=MERGE_AXIS)
    conv7 = conv_bn_relu(filters=64, kernel_size=(3, 3))(up2)
    conv7 = Dropout(0.2)(conv7)
    conv7 = conv_bn_relu(filters=64, kernel_size=(3, 3))(conv7)

    up3 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=MERGE_AXIS)
    conv8 = conv_bn_relu(filters=32, kernel_size=(3, 3))(up3)
    conv8 = Dropout(0.2)(conv8)
    conv8 = conv_bn_relu(filters=32, kernel_size=(3, 3))(conv8)

    up4 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=MERGE_AXIS)
    conv9 = conv_bn_relu(filters=16, kernel_size=(3, 3))(up4)
    conv9 = Dropout(0.2)(conv9)
    conv9 = conv_bn_relu(filters=16, kernel_size=(3, 3))(conv9)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer=optimizers.adam(lr=1e-3, decay=0.1), loss=dice_loss, metrics=[dice_coeff])

    return model


def u_net_bn_alt(input_size):
    """
    A U-Net structure that takes two inputs. input 1 is image and 2 is ROI.
    Concatenate ROI and images after convolution.
    :param input_size: A tuple of input size with 'channel-last' format e.g. (width, height, channel)
    :return:
    """
    input1 = Input(input_size)
    input2 = Input(input_size)
    #Encoding
    conv1 = conv_bn_relu(filters=16, kernel_size=(3, 3))(input1)
    conv1 = Dropout(0.2)(conv1)
    conv1 = conv_bn_relu(filters=16, kernel_size=(3, 3))(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool01 = MaxPooling2D((2, 2))(input2)

    conv2 = conv_bn_relu(filters=32, kernel_size=(3, 3))(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = conv_bn_relu(filters=32, kernel_size=(3, 3))(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool02 = MaxPooling2D((2, 2))(pool01)

    conv3 = conv_bn_relu(filters=64, kernel_size=(3, 3))(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = conv_bn_relu(filters=64, kernel_size=(3, 3))(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool03 = MaxPooling2D((2, 2))(pool02)

    conv4 = conv_bn_relu(filters=128, kernel_size=(3, 3))(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = conv_bn_relu(filters=128, kernel_size=(3, 3))(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool04 = MaxPooling2D((2, 2))(pool03)

    conve = conv_bn_relu(filters=256, kernel_size=(3, 3))(pool4)
    conve = Dropout(0.2)(conve)
    conve = conv_bn_relu(filters=256, kernel_size=(3, 3))(conve)
    poole = MaxPooling2D((2, 2))(conve)

    #Bottom Layer
    conv5 = conv_bn_relu(filters=512, kernel_size=(3, 3))(poole)
    conv5 = Dropout(0.2)(conv5)
    conv5 = conv_bn_relu(filters=512, kernel_size=(3, 3))(conv5)

    # Decoding
    upd = concatenate([UpSampling2D(size=(2, 2))(conv5), conve, pool04], axis=MERGE_AXIS)
    convd = conv_bn_relu(filters=256, kernel_size=(3, 3))(upd)
    convd = Dropout(0.2)(convd)
    convd = conv_bn_relu(filters=256, kernel_size=(3, 3))(convd)

    up1 = concatenate([UpSampling2D(size=(2, 2))(convd), conv4, pool03], axis=MERGE_AXIS)
    conv6 = conv_bn_relu(filters=128, kernel_size=(3, 3))(up1)
    conv6 = Dropout(0.2)(conv6)
    conv6 = conv_bn_relu(filters=128, kernel_size=(3, 3))(conv6)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3, pool02], axis=MERGE_AXIS)
    conv7 = conv_bn_relu(filters=64, kernel_size=(3, 3))(up2)
    conv7 = Dropout(0.2)(conv7)
    conv7 = conv_bn_relu(filters=64, kernel_size=(3, 3))(conv7)

    up3 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2, pool01], axis=MERGE_AXIS)
    conv8 = conv_bn_relu(filters=32, kernel_size=(3, 3))(up3)
    conv8 = Dropout(0.2)(conv8)
    conv8 = conv_bn_relu(filters=32, kernel_size=(3, 3))(conv8)

    up4 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=MERGE_AXIS)
    conv9 = conv_bn_relu(filters=16, kernel_size=(3, 3))(up4)
    conv9 = Dropout(0.2)(conv9)
    conv9 = conv_bn_relu(filters=16, kernel_size=(3, 3))(conv9)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[input1, input2], outputs=[outputs])

    model.compile(optimizer=optimizers.adam(lr=1e-3, decay=0.1), loss=dice_loss, metrics=[dice_coeff])

    return model
