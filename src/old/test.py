from IPython import display

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_probability as tfp
import time
from tensorflow.keras import backend as K

from pathlib import Path

########################################################################

IMAGE_SIZE = (64,64)
EPOCHS = 10
batch_size = 32


# (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

def fetch_data(path):
    # 'path' contains path to
    # directory with the folders
    # 'train_images/' and
    # 'val_images/' 

    # Fetch dataset images
    data = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        color_mode='grayscale',
        batch_size=batch_size,
        image_size=IMAGE_SIZE
    )
    # Compute length
    data_len = len(list(Path(path).rglob("*.*")))
    return (data, data_len)

train_dataset, train_size = fetch_data('data/chinese_mnist/train_images')
test_dataset, test_size = fetch_data('data/chinese_mnist/val_images')

# def preprocess_images(images):
#   images = images.reshape((images.shape[0], IMAGE_SIZE[0], IMAGE_SIZE[1], 1)) / 255.
#   return np.where(images > .5, 1.0, 0.0).astype('float32')

# train_images = preprocess_images(train_images)
# test_images = preprocess_images(test_images)


#train_size = 10500
#test_size = 10000


# train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
#                  .shuffle(train_size).batch(batch_size))
# test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
#                 .shuffle(test_size).batch(batch_size))



class CVAE(tf.keras.Model):

  def __init__(self, latent_dim):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim

    ################ Original ################

    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),

            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )

    ################ Version 1 ################
    
    # self.encoder = tf.keras.Sequential(
    #     [
    #         tf.keras.layers.InputLayer(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),
    #         tf.keras.layers.Conv2D(
    #             filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
    #         tf.keras.layers.Conv2D(
    #             filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
    #     ]
    # )
    # volume_size = K.int_shape(self.encoder)
    # print(volume_size)
    # self.encoder.add(tf.keras.layers.Flatten())
    # self.encoder.add(tf.keras.layers.Dense(latent_dim + latent_dim))

    ################ Version 1 ################

    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),

            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )

    ###########################################

    # encoder_inputs = Input(shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
    # encoder = Conv2D(32, 3, strides=2, padding="same", activation="relu")(encoder_inputs)
    # encoder = Conv2D(64, 3, strides=2, padding="same", activation="relu")(encoder)

    # volume_size = K.int_shape(encoder)
    
    # encoder = Flatten()(encoder)
    # z_mean = Dense(LATENT_DIM)(encoder)
    # z_logvar = Dense(LATENT_DIM)(encoder)
    # z_layer = Sampling()([z_mean, z_logvar])

    # self.encoder = Model(encoder_inputs, [z_layer, z_mean, z_logvar], name="encoder")

    ################ Original ################

    # self.decoder = tf.keras.Sequential(
    #     [
    #         tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
    #         tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
    #         tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
    #         tf.keras.layers.Conv2DTranspose(
    #             filters=64, kernel_size=3, strides=2, padding='same',
    #             activation='relu'),
    #         tf.keras.layers.Conv2DTranspose(
    #             filters=32, kernel_size=3, strides=2, padding='same',
    #             activation='relu'),
    #         # No activation
    #         tf.keras.layers.Conv2DTranspose(
    #             filters=1, kernel_size=3, strides=1, padding='same'),
    #     ]
    # )

    ################ Version 1 ################

    # self.decoder = tf.keras.Sequential(
    #     [
    #         tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
    #         tf.keras.layers.Dense(units=np.prod(volume_size[1:4]), activation=tf.nn.relu),
    #         tf.keras.layers.Reshape(target_shape=volume_size[1:4]),
    #         tf.keras.layers.Conv2DTranspose(
    #             filters=64, kernel_size=3, strides=2, padding='same',
    #             activation='relu'),
    #         tf.keras.layers.Conv2DTranspose(
    #             filters=32, kernel_size=3, strides=2, padding='same',
    #             activation='relu'),
    #         # No activation
    #         tf.keras.layers.Conv2DTranspose(
    #             filters=1, kernel_size=3, strides=1, padding='same'),
    #     ]
    # )
    
    ################ Version 2 ################

    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=16*16*64, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(16, 16, 64)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=1, padding='same'),
        ]
    )

    ###########################################


    @tf.function
    def sample(self, eps=None):
        if eps is None:
          eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


optimizer = tf.keras.optimizers.Adam(1e-4)


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, optimizer):
    """Executes one training step and returns the loss.
  
    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))




# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 2
num_examples_to_generate = 16

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])
model = CVAE(latent_dim)



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


# Pick a sample of the test set for generating output images
assert batch_size >= num_examples_to_generate
for test_batch in test_dataset.take(1):
    test_sample = test_batch[0:num_examples_to_generate, :, :, :]


generate_and_save_images(model, 0, test_sample)

for epoch in range(1, EPOCHS + 1):
    start_time = time.time()
    for train_x in train_dataset:
        train_step(model, train_x, optimizer)
    end_time = time.time()

    loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
        loss(compute_loss(model, test_x))
    elbo = -loss.result()
    display.clear_output(wait=False)
    print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
          .format(epoch, elbo, end_time - start_time))
    generate_and_save_images(model, epoch, test_sample)