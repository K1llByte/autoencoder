#https://www.tensorflow.org/tutorials/generative/cvae

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

###############################################################

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input)
    fig = plt.figure(figsize=(16,256))

    for i in range(predictions.shape[0]):
        plt.subplot(1, 16, i+1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
    
    
    
def plot_scatter(x,y,train_Y):
    cmap = colors.ListedColormap(['black', 'darkred', 'darkblue', 'darkgreen', 'yellow', 'brown', 'purple', 'lightgreen', 'red', 'lightblue'])
    bounds=[0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5,8.5,9.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=(12,10))
    ax = fig.gca()
    ax.set_aspect('equal')
    plt.scatter(x, y, c = train_Y, cmap=cmap, s = 1, norm=norm)
    plt.colorbar()
    plt.gca().add_patch(Rectangle((-2,-2), 4,4, linewidth=2, edgecolor='r', facecolor='none'))   
    plt.show()
    
    
# assumes len samples is a perfect square
def show_samples(samples):
    
    k = int(math.sqrt(len(samples)))
    fig = plt.figure(figsize=(k,k))
    
    for i in range(len(samples)):
        plt.subplot(k, k, i+1)
        plt.imshow(np.asarray(samples)[i, :, :, 0], cmap='gray')
        plt.axis('off')

###############################################################

BATCH_SIZE = 64
LATENT_DIM = 64
EPOCHS = 25
IMAGE_SIZE = (64, 64)

###############################################################

# # load data
# (train_X, train_Y), (test_X, test_Y) = mnist.load_data()

# # reshape and normalize
# train_X = train_X.reshape(train_X.shape[0], 28, 28, 1).astype('float32') / 255.0
# test_X = test_X.reshape(test_X.shape[0], 28, 28, 1).astype('float32') / 255.0

# TEST_BUF = len(test_X)
# TRAIN_BUF = len(train_X)

# # convert numpy to tensors
# train_dataset = Dataset.from_tensor_slices((train_X, train_Y)).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
# test_dataset = Dataset.from_tensor_slices((test_X, test_Y)).batch(BATCH_SIZE)

def convert_path_to_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.keras.preprocessing.image.smart_resize(img, IMAGE_SIZE)
    
    def get_label(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        #return parts[-2] == class_ids
        return int(parts[-2]) - 1

    label = get_label(file_path)
    
    return img, label

def fetch_data(path):
    if path[-1] != '/':
        path += '/'

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # Fetch dataset images
    train_listset = tf.data.Dataset.list_files(f"{path}/train_images/*/*.jpg")
    train_set = train_listset.map(convert_path_to_image, num_parallel_calls=AUTOTUNE)

    test_listset = tf.data.Dataset.list_files(f"{path}/val_images/*/*.jpg")
    test_set = test_listset.map(convert_path_to_image, num_parallel_calls=AUTOTUNE)

    train_set = train_set.cache()
    train_set = train_set.batch(batch_size=BATCH_SIZE)
    train_set = train_set.prefetch(buffer_size=AUTOTUNE)

    test_set = test_set.cache()
    test_set = test_set.batch(batch_size = BATCH_SIZE)
    test_set = test_set.prefetch(buffer_size = AUTOTUNE)
    
    # Compute length
    train_len = len(list(Path(f"{path}/train_images").rglob("*.*")))
    test_len = len(list(Path(f"{path}/val_images").rglob("*.*")))

    return (train_set, train_len) , (test_set, test_len)


(train_dataset, TRAIN_BUF), (test_dataset, TEST_BUF) = fetch_data('data/chinese_mnist/')


# print(list(train_dataset)[0][0].numpy()[0])
# print(np.max(list(train_dataset)[0][0].numpy()[0]))

###############################################################

class Sampling(layers.Layer):

    def call(self, inputs):
        z_mean, z_logvar = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_logvar) * epsilon

###############################################################

encoder_inputs = Input(shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
encoder = Conv2D(32, 3, strides=2, padding = 'same', activation='relu')(encoder_inputs)
encoder = Conv2D(64, 3, strides=2, padding = 'same', activation='relu')(encoder)
encoder = Flatten()(encoder)
z_mean = Dense(LATENT_DIM)(encoder)
z_logvar = Dense(LATENT_DIM)(encoder)
z_layer = Sampling()([z_mean, z_logvar])

encoder = Model(encoder_inputs, [z_layer, z_mean, z_logvar], name = "encoder")

#tf.keras.utils.plot_model(encoder,show_shapes=True)

###############################################################

decoder_inputs = Input(shape = (LATENT_DIM,))
decoder = Dense(units=16*16*64, activation=tf.nn.relu)(decoder_inputs)
decoder = Reshape(target_shape=(16, 16, 64))(decoder)
decoder = Conv2DTranspose(64, 3, strides=2, padding="same",  activation='relu')(decoder)
decoder = Conv2DTranspose(32, 3, strides=2, padding="same",  activation='relu')(decoder)
# reduce to (28,28,1)  
decoder_output = Conv2DTranspose(1, 3, strides=(1, 1), padding="SAME", activation = 'sigmoid')(decoder)

decoder = Model(decoder_inputs, decoder_output)

#tf.keras.utils.plot_model(decoder,show_shapes=True)

###############################################################

vae = Model(encoder_inputs, decoder(z_layer))

###############################################################

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

###############################################################

num_examples_to_generate = 16
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, LATENT_DIM])

generate_and_save_images(
            decoder, EPOCHS, random_vector_for_generation)

###############################################################

# optimizer = tf.keras.optimizers.Adam(lr = 1e-3)

# for epoch in range(1, EPOCHS + 1):
    
#     epoch_loss = 0
#     rec_loss = 0
#     kls_loss = 0
#     batch = 0
    
#     for train_x, _ in train_dataset:
        
#         with tf.GradientTape() as tape:
#             # Compute Loss            
#             z, z_mean, z_logvar = encoder(train_x)
#             x = decoder(z)
            
#             kl_loss = -0.5 * (1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar))
#             kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))  

#             reconstruction_loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(train_x, x), axis=(1, 2))
#             reconstruction_loss = tf.reduce_mean(reconstruction_loss)        
            
#             loss = kl_loss + reconstruction_loss
            
#         gradients = tape.gradient(loss, vae.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
        
#         epoch_loss += tf.reduce_mean(loss)
#         kls_loss += tf.reduce_mean(kl_loss)
#         rec_loss += tf.reduce_mean(reconstruction_loss)
#         batch += 1
#     print(f'Epoch {epoch}, loss: {epoch_loss/batch}, reconst loss: {rec_loss/batch}, KL loss: {kls_loss/batch}')
#     if epoch % 10 == 0:
#         generate_and_save_images(
#             decoder, epoch, random_vector_for_generation)

###############################################################

#vae.save_weights('vae.weights')
vae.load_weights('vae.weights')

###############################################################

samples = []

for train_x, _ in train_dataset:
    z, mean, logvar = encoder(train_x)
    samples.extend(decoder(z).numpy())
    
show_samples(samples[:100])

###############################################################

# vector_generation = []
# for i in range(20):
#     for j in range(20):
#         vector_generation.append([-2.0 + i*0.2, -2.0 + j*0.2])

# predictions = decoder(np.asarray(vector_generation))

# show_samples(predictions)