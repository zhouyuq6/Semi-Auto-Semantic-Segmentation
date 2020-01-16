import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard
from u_net import *
from preprocessing import *
from load_data import *
from predict import *
from preprocessing import *

# GPU ALLOCATION
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
# FILE PATH
data_path = r'/home/zhouyuqian/Kidney_data/'
target_path = r'/home/zhouyuqian/kid_data'
log_path = r'/home/zhouyuqian/logs'

# LOAD DATA
train_case_id, val_case_id, test_case_id = index_split(150, 0, 0.2, 0.1)
print(train_case_id, val_case_id, test_case_id)

# GENERATE DATASET
for cid in train_case_id:
    load_data_from_case(cid, data_path, target_path+'/train')

for cid in val_case_id:
    load_data_from_case(cid, data_path, target_path+'/val')

for cid in test_case_id:
    load_data_from_case(cid, data_path, target_path+'/test')

# TRAIN
model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
tensor_board = TensorBoard(log_dir=log_path)
model = u_net_bn((256, 256, 2))
print(model.summary())

# PARAMETERS
batch_size = 20
train_num = 15538
train_steps = int(train_num/batch_size)
val_num = 5064
val_steps = int(val_num/batch_size)

# FIT
model_history = model.fit_generator(
    data_generator(target_path+'/train', batch_size=batch_size),
    steps_per_epoch=train_steps,
    epochs=100,
    verbose=1,
    callbacks=[model_checkpoint, tensor_board],
    validation_data=test_generator(target_path+'/val', batch_size=batch_size),
    validation_steps=val_steps,
)
