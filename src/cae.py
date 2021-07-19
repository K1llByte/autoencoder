# imports
import os
import pathlib
import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

#from utils import *

# Configuration
HEIGHT = 64
WIDTH = 64
BATCH_SIZE = 8
LATENT_SPACE_DIM = 64
EPOCHS = 25

def show_samples(samples):
    import numpy as np
    k = int(math.sqrt(len(samples)))
    fig = plt.figure(figsize=(k,k))
    
    for i in range(len(samples)):
        plt.subplot(k, k, i+1)
        plt.imshow(np.asarray(samples)[i, :, :, 0], cmap='gray')
        plt.axis('off')

def convert_path_to_image(file_path):
  img = tf.io.read_file(file_path)
  img = tf.image.decode_png(img, channels=1)
  img = tf.image.convert_image_dtype(img, tf.float32)
  img = tf.keras.preprocessing.image.smart_resize(img, [HEIGHT,WIDTH])
  return img, img

def fetch_data(path="data/chinese_mnist"):

    ################################ Load Dataset ##################################

    print("[INFO] Fetching data ...")
    if path[-1] != '/':
        path += '/'

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_listset = tf.data.Dataset.list_files(f"{path}/train_images/*/*.jpg")
    train_set = train_listset.map(convert_path_to_image, num_parallel_calls=AUTOTUNE)

    val_listset = tf.data.Dataset.list_files(f"{path}/val_images/*/*.jpg")
    val_set = val_listset.map(convert_path_to_image, num_parallel_calls=AUTOTUNE)

    ############################### Prepare Dataset ################################

    train_set = train_set.cache()
    train_set = train_set.batch(batch_size=BATCH_SIZE)
    train_set = train_set.prefetch(buffer_size=AUTOTUNE)

    val_set = val_set.cache()
    val_set = val_set.batch(batch_size = BATCH_SIZE)
    val_set = val_set.prefetch(buffer_size = AUTOTUNE)

    return train_set, val_set

def make_model(num_channels=1):
    print("[INFO] Make model ...")

    # Encoder
    inputs = layers.Input(shape =(HEIGHT, WIDTH, num_channels))

    x = layers.Conv2D(32, (3, 3), strides=2, padding="same")(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(48, (3, 3), strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(64, (3, 3), strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization()(x)

    volume_size = tf.keras.backend.int_shape(x)
    x = layers.Flatten()(x)

    # Latent space
    latent = layers.Dense(LATENT_SPACE_DIM, name="latent")(x)

    #decoder
    latent_inputs = layers.Input(shape=(LATENT_SPACE_DIM,))
    y = layers.Dense(np.prod(volume_size[1:4]))(latent_inputs)
    y = layers.Reshape(volume_size[1:4])(y)

    y = layers.Conv2DTranspose(64, (3, 3), strides=2, padding="same")(y)
    y = layers.LeakyReLU(alpha=0.2)(y)
    y = layers.BatchNormalization()(y)

    y = layers.Conv2DTranspose(48, (3, 3), strides=2, padding="same")(y)
    y = layers.LeakyReLU(alpha=0.2)(y)
    y = layers.BatchNormalization()(y)

    y = layers.Conv2DTranspose(32, (3, 3), strides=2, padding="same")(y)
    y = layers.LeakyReLU(alpha=0.2)(y)
    y = layers.BatchNormalization()(y)

    y = layers.Conv2DTranspose(num_channels, (3, 3), padding="same")(y)
    outputs = layers.Activation("sigmoid", name="decoded")(y)

    #encoder = Model(input = x, outupt = latent)
    encoder = Model(inputs, latent, name="encoder")
    decoder = Model(latent_inputs, outputs, name="decoder")
    autoencoder = Model(inputs=inputs, outputs=decoder(encoder(inputs)))

    # encoder.summary()
    # decoder.summary()
    # autoencoder.summary()

    # compile model
    autoencoder.compile(loss="mse", optimizer=Adam(learning_rate=1e-3), metrics=['accuracy'])

    return encoder, decoder, autoencoder


def train(in_model, data, epochs, save_path='models'):
    data_set, val_set = data
    encoder, decoder, autoencoder = in_model
    if save_path[-1] == '/':
        save_path = save_path[:-1]

    model_dir = f"ae_{EPOCHS}e_{LATENT_SPACE_DIM}l_{WIDTH}w"
    autoencoder_file = f"{save_path}/{model_dir}/autoencoder.h5"
    encoder_file = f"{save_path}/{model_dir}/encoder.h5"
    decoder_file = f"{save_path}/{model_dir}/decoder.h5"

    if not os.path.exists(f"{save_path}/{model_dir}/"):
        print("[INFO] Model wans't found, generating ...")

        os.mkdir(model_dir)

        # Train autoencoder
        print("[INFO] Training Model ...")
        autoencoder.fit(
                data_set,
                validation_data=val_set,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE)

        print("[INFO] Training Finished")
        encoder.save(encoder_file)
        decoder.save(decoder_file)
        autoencoder.save(autoencoder_file)
        print("[INFO] Saved Trained Models")

    else:
        print("[INFO] Trying to load pretrained model")
        encoder = tf.keras.models.load_model(encoder_file)
        decoder = tf.keras.models.load_model(decoder_file)
        autoencoder = tf.keras.models.load_model(autoencoder_file)
        #autoencoder.compile(loss="mse", optimizer=Adam(learning_rate=1e-3), metrics=['accuracy'])
        print("[INFO] Loaded Trained Model")

    return (encoder, decoder, autoencoder)

def predict_all(in_model, in_data):
    encoder, decoder, autoencoder = in_model
    data_set, val_set = in_data

    
    print("[INFO] Predicting set")
    preds = autoencoder.predict(data_set)

    return preds

#####################################################################

if __name__ == '__main__':
    # Fetch and prepare data
    data = fetch_data("data/chinese_mnist")

    # Make and compile model
    model = make_model()

    # Train or load model
    model = train(model, data, EPOCHS, save_path='models')

    # Get predictions for dataset
    preds = predict_all(model, data)

    print("[INFO] Displaying some results")
    def display_from_dataset(i):
        data_set = list(data[0])

        (a,b) = divmod(i,BATCH_SIZE)

        original = data_set[a][1].numpy()[b]
        predicted = preds[i]

        fig, axarr = plt.subplots(1,2)
        axarr[0].imshow(original)
        axarr[0].set_title("original")
        axarr[0].axis('off')
        axarr[1].imshow(predicted)
        axarr[1].set_title("predicted")
        axarr[1].axis('off')

        fig.tight_layout()
        plt.show()