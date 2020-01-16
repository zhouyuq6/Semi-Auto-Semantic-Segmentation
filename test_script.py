import os
import tensorflow as tf
from u_net import *
from preprocessing import *
from load_data import *
from predict import *

# GPU ALLOCATION
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
# FILE PATH
data_path = r'/home/zhouyuqian/Kidney_data/'
target_path = r'/home/zhouyuqian/kid_data_og/'
weight_path = r'weights.h5'
# TEST CASE ID
test_id = [15, 17, 19, 24, 33, 41, 44, 48, 52, 56, 66, 73, 87, 94, 95, 105, 106, 107, 109, 112, 113, 115, 116, 132,
           134, 139, 140, 142, 144, 147]

# PREDICT
model = u_net_bn((256, 256, 2))
A_loss, A_dsc = predict_by_slice(test_id, model=model, data_path=data_path, weights=weight_path)
print("A: Test Loss:", A_loss, "Test Dice:", A_dsc)

B_loss, B_dsc = predict_from_roi(test_id, model=model, data_path=data_path, weights=weight_path)
print("B: Test Loss:", B_loss, "Test Dice:", B_dsc)

C_loss, C_dsc = predict_from_self(test_id, model=model, data_path=data_path, weights=weight_path)
print("C: Test Loss:", C_loss, "Test Dice:", C_dsc)
