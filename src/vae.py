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

#from utils import *

###############################################################

BATCH_SIZE = 64
LATENT_DIM = 64
EPOCHS = 25
IMAGE_SIZE = (64, 64)

###############################################################

def show_samples(samples):
    import numpy as np
    k = int(math.sqrt(len(samples)))
    fig = plt.figure(figsize=(k,k))
    
    for i in range(len(samples)):
        plt.subplot(k, k, i+1)
        plt.imshow(np.asarray(samples)[i, :, :, 0], cmap='gray')
        plt.axis('off')
        
# def generate_and_save_images(model, epoch, test_input):
#     predictions = model(test_input)
#     fig = plt.figure(figsize=(16,256))

#     for i in range(predictions.shape[0]):
#         plt.subplot(1, 16, i+1)
#         plt.imshow(predictions[i, :, :, 0], cmap='gray')
#         plt.axis('off')

#     # tight_layout minimizes the overlap between 2 sub-plots
#     plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
#     plt.show()
    
    
    
# def plot_scatter(x,y,train_Y):
#     cmap = colors.ListedColormap(['black', 'darkred', 'darkblue', 'darkgreen', 'yellow', 'brown', 'purple', 'lightgreen', 'red', 'lightblue'])
#     bounds=[0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5,8.5,9.5]
#     norm = colors.BoundaryNorm(bounds, cmap.N)

#     fig = plt.figure(figsize=(12,10))
#     ax = fig.gca()
#     ax.set_aspect('equal')
#     plt.scatter(x, y, c = train_Y, cmap=cmap, s = 1, norm=norm)
#     plt.colorbar()
#     plt.gca().add_patch(Rectangle((-2,-2), 4,4, linewidth=2, edgecolor='r', facecolor='none'))   
#     plt.show()
    
    
# # assumes len samples is a perfect square
# def show_samples(samples):
    
#     k = int(math.sqrt(len(samples)))
#     fig = plt.figure(figsize=(k,k))
    
#     for i in range(len(samples)):
#         plt.subplot(k, k, i+1)
#         plt.imshow(np.asarray(samples)[i, :, :, 0], cmap='gray')
#         plt.axis('off')

###############################################################

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

###############################################################

class Sampling(layers.Layer):

    def call(self, inputs):
        z_mean, z_logvar = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_logvar) * epsilon

    def get_config(self):
        config = super(Sampling, self).get_config()
        return config

###############################################################

def make_model():

    encoder_inputs = Input(shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
    encoder = Conv2D(32, 3, strides=2, padding = 'same', activation='relu')(encoder_inputs)
    encoder = Conv2D(64, 3, strides=2, padding = 'same', activation='relu')(encoder)
    encoder = Flatten()(encoder)
    z_mean = Dense(LATENT_DIM)(encoder)
    z_logvar = Dense(LATENT_DIM)(encoder)
    z_layer = Sampling()([z_mean, z_logvar])

    encoder = Model(encoder_inputs, [z_layer, z_mean, z_logvar], name = "encoder")

    ###############################################################

    decoder_inputs = Input(shape = (LATENT_DIM,))
    decoder = Dense(units=16*16*64, activation=tf.nn.relu)(decoder_inputs)
    decoder = Reshape(target_shape=(16, 16, 64))(decoder)
    decoder = Conv2DTranspose(64, 3, strides=2, padding="same",  activation='relu')(decoder)
    decoder = Conv2DTranspose(32, 3, strides=2, padding="same",  activation='relu')(decoder)
    # reduce to (28,28,1)
    decoder_output = Conv2DTranspose(1, 3, strides=(1, 1), padding="SAME", activation = 'sigmoid')(decoder)

    decoder = Model(decoder_inputs, decoder_output, name = "decoder")

    ###############################################################

    vae = Model(encoder_inputs, decoder(z_layer))

    return encoder, decoder, vae

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

# num_examples_to_generate = 16
# random_vector_for_generation = tf.random.normal(
#     shape=[num_examples_to_generate, LATENT_DIM])

# generate_and_save_images(
#             decoder, EPOCHS, random_vector_for_generation)

###############################################################

def train(in_model, data, epochs, save_path="models"):

    encoder, decoder, vae = in_model
    (train_dataset, train_len), (test_dataset, test_len) = data

    model_dir = f"vae_{epochs}e_{LATENT_DIM}l_{IMAGE_SIZE[0]}w"
    # autoencoder_file = f"{save_path}/{model_dir}/autoencoder.h5"
    # encoder_file = f"{save_path}/{model_dir}/encoder.h5"
    # decoder_file = f"{save_path}/{model_dir}/decoder.h5"

    autoencoder_file = f"{save_path}/{model_dir}/autoencoder.weights"

    if not os.path.exists(f"{save_path}/{model_dir}/"):
        print("[INFO] Model wans't found, generating ...")

        os.mkdir(f"{save_path}/{model_dir}")

        # Train variational autoencoder
        print("[INFO] Training Model ...")

        optimizer = tf.keras.optimizers.Adam(lr = 1e-3)

        for epoch in range(1, epochs + 1):

            epoch_loss = 0
            rec_loss = 0
            kls_loss = 0
            batch = 0

            for train_x, _ in train_dataset:

                with tf.GradientTape() as tape:

                    # Compute Loss
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

        print("[INFO] Training Finished")
        # encoder.save(encoder_file)
        # decoder.save(decoder_file)
        # vae.save(autoencoder_file)

        vae.save_weights(f"{save_path}/{model_dir}/vae.weights")
        print("[INFO] Saved Trained Models")

    else:
        print("[INFO] Trying to load pretrained model")
        # encoder = tf.keras.models.load_model(encoder_file)
        # decoder = tf.keras.models.load_model(decoder_file)
        # vae = tf.keras.models.load_model(autoencoder_file)

        vae.load_weights(f"{save_path}/{model_dir}/vae.weights")
        print("[INFO] Loaded Trained Model")

    return (encoder, decoder, vae)

def predict_all(in_model, in_data):
    encoder, decoder, autoencoder = in_model
    (data_set, data_len), (val_set, val_len) = in_data

    
    print("[INFO] Predicting set")
    preds = autoencoder.predict(data_set)

    return preds

###############################################################

if __name__ == "__main__":
    # Fetch and prepare data
    data = fetch_data('data/chinese_mnist/')
    
    # Make and compile model
    model = make_model()
    
    # Train or load model
    model = train(model, data, EPOCHS)

    encoder, decoder, vae = model
    (train_dataset, train_len), (test_dataset, test_len) = data

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