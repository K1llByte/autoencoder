import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Flatten, Dense, Reshape, Input, InputLayer
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.data import Dataset

import os
import numpy as np
import math

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle

##############################################################

BATCH_SIZE = 64
LATENT_DIM = 2 # 64

(train_X, train_Y), (test_X, test_Y) = mnist.load_data()

print(train_X[0])


# train_data = tf.keras.preprocessing.image_dataset_from_directory('data/chinese_mnist/train_images', batch_size=BATCH_SIZE, image_size=(128, 128))

# Dataset.from_tensor_slices((train_X, train_Y)).shuffle(TRAIN_BUF).batch(BATCH_SIZE)

# train_set_len = tf.data.experimental.cardinality(train_data).numpy()
# print(train_set_len)

#print(len(list(train_data)[0][0].numpy()))

# class chinese_mnist:
#     # 'path' contains path to
#     # directory with the folders
#     # 'train_images/' and
#     # 'val_images/' 
#     def load_data(path):
#         pass