import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.utils import plot_model
from scipy.linalg import sqrtm
from skimage.transform import resize
import matplotlib.pyplot as plt
import os
import math
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint

# Set the dimensions for our images
img_rows, img_cols, channels = 28, 28, 1
img_shape = (img_rows, img_cols, channels)
z_dim = 100


# Generator
def build_generator(z_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=z_dim))  # Increased to 256 units
    model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    model.add(Dense(512)) # Additional layer
    model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    model.add(Dense(img_rows * img_cols * channels, activation='tanh'))
    model.add(Reshape(img_shape))
    model.summary()
    return model

# Discriminator
def build_discriminator(img_shape):
    model = Sequential()
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.02))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    return model

# Save generated images
def save_images(epoch, generator, z_dim, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.normal(0, 1, (examples, z_dim))
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    if not os.path.exists('images/'):
        os.makedirs('images/')
    plt.savefig(f'images/gan_generated_image_epoch_{epoch}.png')

# Build and compile the Discriminator
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

# Build the Generator
generator = build_generator(z_dim)

# Keep Discriminator's parameters constant for Generator training
discriminator.trainable = False

# Build and compile the combined model
gan = Sequential([generator, discriminator])
gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0004))

# Define the learning rate scheduler
def step_decay(epoch):
    initial_lrate = 0.0004
    drop = 0.0001
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

# LearningRateScheduler callback
lrate = LearningRateScheduler(step_decay)

# Ensure model directories exist
if not os.path.exists('models/'):
    os.makedirs('models/')

# ModelCheckpoint callbacks for saving models
checkpoint_gen = ModelCheckpoint('models/generator_epoch_{epoch:02d}.keras', save_freq='epoch')
checkpoint_disc = ModelCheckpoint('models/discriminator_epoch_{epoch:02d}.keras', save_freq='epoch')

# Training function
def train_gan(gan, generator, discriminator, img_shape, z_dim, epochs=37000, batch_size=512, save_interval=100):
    (X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    X_train = X_train / 127.5 - 1.0
    X_train = np.expand_dims(X_train, axis=3)

    smooth_factor = 0.9
    real = np.ones((batch_size, 1)) * smooth_factor
    fake = np.zeros((batch_size, 1))

    d_losses, g_losses = [], []

    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict(z)

        try:
            d_loss_real = discriminator.train_on_batch(imgs, real)
            d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        except Exception as e:
            print(f"Error in discriminator training at epoch {epoch}: {e}")
            continue

        z = np.random.normal(0, 1, (batch_size, z_dim))
        try:
            g_loss = gan.train_on_batch(z, real)
        except Exception as e:
            print(f"Error in generator training at epoch {epoch}: {e}")
            continue

        d_losses.append(d_loss)
        g_losses.append(g_loss)

        if epoch % save_interval == 0:
            print(f"Epoch: {epoch}, D Loss: {d_loss}, G Loss: {g_loss}")
            save_images(epoch, generator, z_dim)
            generator.save(f'models/generator_epoch_{epoch}.keras')
            discriminator.save(f'models/discriminator_epoch_{epoch}.keras')

    plt.figure(figsize=(10,5))
    plt.plot(d_losses, label="Discriminator Loss")
    plt.plot(g_losses, label="Generator Loss")
    plt.title("Training Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('training_losses.png')
    plt.show()

# Start training
train_gan(gan, generator, discriminator, img_shape, z_dim)

def generate_images(generator, z_dim, num_imgs=10):
    # Generate noise vectors as input for GAN
    z = np.random.normal(0, 1, (num_imgs, z_dim))

    # Generate images from noise vectors
    gen_imgs = generator.predict(z)

    # Rescale images to [0, 1]
    gen_imgs = 0.5 * gen_imgs + 0.5

    # Plot generated images
    fig, axs = plt.subplots(1, num_imgs, figsize=(20, 2))
    for i in range(num_imgs):
        axs[i].imshow(gen_imgs[i, :, :, 0], cmap='gray')
        axs[i].axis('off')
    plt.show()

# Generate and display images
generate_images(generator, z_dim)
