################################### Imports ###################################

import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Conv2D, MaxPooling2D, Dense, Activation, Flatten, Dropout
from tensorflow.keras.layers import Conv2DTranspose, Reshape, Input, InputLayer
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.data import Dataset
from tensorflow.keras import backend as K

import os
import pathlib
import math
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import IPython.display as display

BATCH_SIZE = 32
IMAGE_SIZE = 64
LATENT_DIM = 2
#class_ids = np.array(['00000','00001', '00002', '00003', '00004', '00005', '00006', '00007'])
#class_names = ['Limit 20', 'Limit 30', 'Limit 51', 'Limit 60', 'Limit 70', 'Limit 80', 'Limit 100', 'Limit 120']
class_ids = np.array(os.listdir(f"../data/chinese_mnist/train_images"))
class_names = class_ids
NUM_CLASSES = len(class_ids)

############################## Auxiliar Functions #############################

def get_bytes(file_path):
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=1)
    
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)

    # Resize the image to the desired size (No aspect ratio distortion).
    img = tf.keras.preprocessing.image.smart_resize(img, [IMAGE_SIZE,IMAGE_SIZE])
    return img, img

################################### Dataset ###################################

def fetch_data(path="data/chinese_mnist"):

    ################################ Load Dataset ##################################

    if path[-1] != '/':
        path += '/'

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_listset = tf.data.Dataset.list_files(f"{path}/train_images/*/*.jpg")
    train_set = train_listset.map(get_bytes, num_parallel_calls=AUTOTUNE)

    val_listset = tf.data.Dataset.list_files(f"{path}/val_images/*/*.jpg")
    val_set = val_listset.map(get_bytes, num_parallel_calls=AUTOTUNE)

    ############################### Prepare Dataset ################################

    train_set_len = tf.data.experimental.cardinality(train_set).numpy()
    val_set_len = tf.data.experimental.cardinality(val_set).numpy()
    
    train_set = train_set.cache()
    train_set = train_set.shuffle(buffer_size=train_set_len)
    train_set = train_set.batch(batch_size=BATCH_SIZE)
    train_set = train_set.prefetch(buffer_size=AUTOTUNE)
    # train_set = train_set.repeat()

    val_set = val_set.cache()
    val_set = val_set.shuffle(buffer_size=val_set_len)
    val_set = val_set.batch(batch_size = BATCH_SIZE)
    val_set = val_set.prefetch(buffer_size = AUTOTUNE)
    # val_set = val_set.repeat()


    # testset_length = [i for i,_ in enumerate(test_set)][-1] + 1
    # print('Number of batches: ', testset_length)
    
    #train_set = np.expand_dims(train_set, axis=-1)
    #val_set = np.expand_dims(val_set, axis=-1)

    return train_set, val_set, train_set_len

#################################### Model ####################################

################################ Define Model #################################

def make_model(class_count, img_size, channels=1):

    inputs = Input(shape=(img_size, img_size, channels))
    
    # Encoder    
    x = Conv2D(32, (3, 3), strides=2, padding="same")(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)

    volumeSize = K.int_shape(x)
    x = Flatten()(x)

    # Latent space
    latent = Dense(LATENT_DIM, name="latent")(x)

    #decoder
    latentInputs = Input(shape=(LATENT_DIM,))
    y = Dense(np.prod(volumeSize[1:]))(latentInputs)
    y = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(y)

    y = Conv2DTranspose(128, (3, 3), strides=2, padding="same")(y)
    y = LeakyReLU(alpha=0.2)(y)
    y = BatchNormalization()(y) 

    y = Conv2DTranspose(64, (3, 3), strides=2, padding="same")(y)
    y = LeakyReLU(alpha=0.2)(y)
    y = BatchNormalization()(y)  

    y = Conv2DTranspose(32, (3, 3), strides=2, padding="same")(y)
    y = LeakyReLU(alpha=0.2)(y)
    y = BatchNormalization()(y)  

    y = Conv2DTranspose(channels, (3, 3), padding="same")(y)
    outputs = Activation("sigmoid", name="decoded")(y)

    #encoder = Model(input = x, outupt = latent)
    encoder = Model(inputs, latent, name="encoder")
    decoder = Model(latentInputs, outputs, name="decoder")

    model = Model(inputs=inputs, outputs=decoder(encoder(inputs)))

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='mse')
    return encoder, decoder, model


def train(in_model, data, model_file='models/autoencoder',num_epochs=20):
    train_set, val_set, dataset_length = data
    steps = math.ceil(dataset_length * 0.3)/BATCH_SIZE



    if not os.path.exists(model_file):
        print("[INFO] Training Model ...")

        # print("train_set",train_set)
        # print("val_set",val_set)
        
        in_model[2].fit(train_set,
                    #steps_per_epoch=dataset_length/BATCH_SIZE,
                    epochs=num_epochs,
                    validation_data=val_set,
                    batch_size=BATCH_SIZE
                    #validation_steps=steps
                    )
        print("[INFO] Training Finished")

        #values = in_model.evaluate(test_set, verbose=1)

        os.mkdir(model_file)
        in_model[0].save(f'{model_file}/encoder.h5')
        in_model[1].save(f'{model_file}/decoder.h5')
        in_model[2].save(f'{model_file}/autoencoder.h5')

        
    else:
        in_model[0] = tf.keras.models.load_model(f'{model_file}/encoder.h5')
        in_model[1] = tf.keras.models.load_model(f'{model_file}/decoder.h5')
        in_model[2] = tf.keras.models.load_model(f'{model_file}/autoencoder.h5')
        print("[INFO] Loaded Trained Model")

    return in_model

def latent_predict(in_model, latent_params):
    _, decoder, _ = in_model
    return decoder.predict(latent_params)

def load_and_predict(in_model, data):
    _, _, autoencoder = in_model
    print("ok1")
    res = autoencoder.predict(data[1])
    print("ok2")
    return res


    # tmp = list(data[1])
    # img = tmp[0]
    # print(img)
    # preds = in_model.predict([img])
    # pred = list(preds)[0]
    # return img, pred