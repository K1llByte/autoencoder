import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Flatten, Dense, Reshape, Input, InputLayer
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.data import Dataset

import os
import numpy as np
import math

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle

from pathlib import Path

from utils import *

##############################################################

BATCH_SIZE = 64
LATENT_DIM = 2 # 64
IMAGE_SIZE = (64, 64)
EPOCHS = 10

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
        image_size=IMAGE_SIZE
    )
    # Compute length
    data_len = len(list(Path(path).rglob("*.*")))
    return (data, data_len)


train_data, train_len = fetch_data('data/chinese_mnist/train_images')
#val_data, val_len = fetch_data('data/chinese_mnist/val_images')


# class CVAE(tf.keras.Model):
#   """Convolutional variational autoencoder."""

#   def __init__(self, latent_dim):
#     super(CVAE, self).__init__()
#     self.latent_dim = latent_dim
#     self.encoder = tf.keras.Sequential(
#         [
#             tf.keras.layers.InputLayer(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),
#             tf.keras.layers.Conv2D(
#                 filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
#             tf.keras.layers.Conv2D(
#                 filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
#             tf.keras.layers.Flatten(),
#             # No activation
#             tf.keras.layers.Dense(latent_dim + latent_dim),
#         ]
#     )

#     self.decoder = tf.keras.Sequential(
#         [
#             tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
#             tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
#             tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
#             tf.keras.layers.Conv2DTranspose(
#                 filters=64, kernel_size=3, strides=2, padding='same',
#                 activation='relu'),
#             tf.keras.layers.Conv2DTranspose(
#                 filters=32, kernel_size=3, strides=2, padding='same',
#                 activation='relu'),
#             # No activation
#             tf.keras.layers.Conv2DTranspose(
#                 filters=1, kernel_size=3, strides=1, padding='same'),
#         ]
#     )

#   @tf.function
#   def sample(self, eps=None):
#     if eps is None:
#       eps = tf.random.normal(shape=(100, self.latent_dim))
#     return self.decode(eps, apply_sigmoid=True)

#   def encode(self, x):
#     mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
#     return mean, logvar

#   def reparameterize(self, mean, logvar):
#     eps = tf.random.normal(shape=mean.shape)
#     return eps * tf.exp(logvar * .5) + mean

#   def decode(self, z, apply_sigmoid=False):
#     logits = self.decoder(z)
#     if apply_sigmoid:
#       probs = tf.sigmoid(logits)
#       return probs
#     return logits


    
class Sampling(layers.Layer):

    def call(self, inputs):
        z_mean, z_logvar = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_logvar) * epsilon

encoder = None
decoder = None

def make_model():

    global encoder
    global decoder

    # Encoder
    encoder_inputs = Input(shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
    encoder = Conv2D(32, 3, strides=2, padding="same", activation="relu")(encoder_inputs)
    encoder = Conv2D(64, 3, strides=2, padding="same", activation="relu")(encoder)
    volume_size = K.int_shape(encoder)
    #print(volume_size)
    encoder = Flatten()(encoder)
    z_mean = Dense(LATENT_DIM)(encoder)
    z_logvar = Dense(LATENT_DIM)(encoder)
    z_layer = Sampling()([z_mean, z_logvar])

    encoder = Model(encoder_inputs, [z_layer, z_mean, z_logvar], name="encoder")

    # Decoder
    decoder_inputs = Input(shape = (LATENT_DIM,))
    decoder = Dense(np.prod(volume_size[1:]), activation=tf.nn.relu)(decoder_inputs)
    decoder = Reshape(target_shape=volume_size[1:4])(decoder)
    decoder = Conv2DTranspose(64, 3, strides=2, padding="same",  activation="relu")(decoder)
    decoder = Conv2DTranspose(32, 3, strides=2, padding="same",  activation="relu")(decoder)
    # reduce to (28,28,1)  
    decoder_output = Conv2DTranspose(1, 3, strides=(1, 1), padding="same", activation="sigmoid")(decoder)

    decoder = Model(decoder_inputs, decoder_output)

    vae = Model(encoder_inputs, decoder(z_layer))

    return vae

vae = make_model()


# def reparameterize(self, mean, logvar):
#     eps = tf.random.normal(shape=mean.shape)
#     return eps * tf.exp(logvar * .5) + mean

@tf.function
def compute_loss(model, data):
    
    z_mean, z_logvar = encoder(data)
    z = reparameterize(z_mean, z_logvar)
    x = decoder(z)
          
    kl_loss = -0.5 * (1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar))
    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))  
    
    reconstruction_loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(data, x), axis=(1, 2))
    reconstruction_loss = tf.reduce_mean(reconstruction_loss)        
     
    return reconstruction_loss, kl_loss, reconstruction_loss  + kl_loss

def generate_and_save_images(model, epoch, test_sample):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()



####### TRAINING #######


optimizer = tf.keras.optimizers.Adam(lr = 1e-3)

for epoch in range(1, EPOCHS + 1):
    
    epoch_loss = 0
    rec_loss = 0
    kls_loss = 0
    batch = 0
    
    for train_x, _ in train_data:
        
        with tf.GradientTape() as tape:
            
            z, z_mean, z_logvar = encoder(train_x)
            x = decoder(z)
            
            kl_loss = -0.5 * (1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))  

            reconstruction_loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(train_x, x), axis=(1, 2))
            reconstruction_loss = tf.reduce_mean(reconstruction_loss)        
            
            loss = kl_loss + reconstruction_loss
            
        gradients = tape.gradient(loss, vae.trainable_variables)
        optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
        
        epoch_loss += tf.reduce_mean(loss)
        kls_loss += tf.reduce_mean(kl_loss)
        rec_loss += tf.reduce_mean(reconstruction_loss)
        batch += 1
    print(f'Epoch {epoch}, loss: {epoch_loss/batch}, reconst loss: {rec_loss/batch}, KL loss: {kls_loss/batch}')
    # if epoch % 10 == 0:
    #     generate_and_save_images(
    #         decoder, epoch, random_vector_for_generation)


# Save weights

SAVE_FILE = "models/cvae.weights"
if os.path.exists(SAVE_FILE):
    vae.load_weights(SAVE_FILE)
else:
    vae.save_weights(SAVE_FILE)


#################################################

samples = []

for train_x, _ in train_data:
    z, _, _ = encoder(train_x)
    samples.extend(decoder(z).numpy())
    
show_samples(samples[:100])  