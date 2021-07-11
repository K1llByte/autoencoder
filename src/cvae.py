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

from pathlib import Path

##############################################################

BATCH_SIZE = 64
LATENT_DIM = 2 # 64


def fetch_data(path):
    # 'path' contains path to
    # directory with the folders
    # 'train_images/' and
    # 'val_images/' 

    # Fetch dataset images
    data = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        image_size=(128, 128)
    )
    # Compute length
    data_len = len(list(Path(path).rglob("*.*")))
    return (data, data_len)


train_data, train_len = fetch_data('data/chinese_mnist/train_images')



class Sampling(layers.Layer):

    def call(self, inputs):
        z_mean, z_logvar = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_logvar) * epsilon

def make_model():

    # Encoder
    encoder_inputs = Input(shape = (28, 28, 1))
    encoder = Conv2D(32, 3, strides=2, padding='same', activation='relu')(encoder_inputs)
    encoder = Conv2D(64, 3, strides=2, padding='same', activation='relu')(encoder)
    encoder = Flatten()(encoder)
    z_mean = Dense(LATENT_DIM)(encoder)
    z_logvar = Dense(LATENT_DIM)(encoder)
    z_layer = Sampling()([z_mean, z_logvar])

    encoder = Model(encoder_inputs, [z_layer, z_mean, z_logvar], name="encoder")

    # Decoder
    decoder_inputs = Input(shape = (LATENT_DIM,))
    decoder = Dense(units=7*7*32, activation=tf.nn.relu)(decoder_inputs)
    decoder = Reshape(target_shape=(7, 7, 32))(decoder)
    decoder = Conv2DTranspose(64, 3, strides=2, padding="same",  activation='relu')(decoder)
    decoder = Conv2DTranspose(32, 3, strides=2, padding="same",  activation='relu')(decoder)
    # reduce to (28,28,1)  
    decoder_output = Conv2DTranspose(1, 3, strides=(1, 1), padding="SAME", activation = 'sigmoid')(decoder)

    decoder = Model(decoder_inputs, decoder_output)

    vae = Model(encoder_inputs, decoder(z_layer))