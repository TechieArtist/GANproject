import numpy as np
import tensorflow as tf
from data.data_loader import load_mnist
from models.generator import build_generator
from models.discriminator import build_discriminator
from models.gan import build_gan
from utils.visualization import save_images
from utils.checkpoint import create_checkpoint_callback
import matplotlib.pyplot as plt
import config

def train_gan(epochs, batch_size, z_dim, img_shape):
    X_train = load_mnist()

    discriminator = build_discriminator(img_shape)
    discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

    generator = build_generator(z_dim, img_shape)

    discriminator.trainable = False
    gan = build_gan(generator, discriminator)
    gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0004))

    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    d_losses, g_losses = [], []

    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        noise = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, z_dim))
        g_loss = gan.train_on_batch(noise, real)

        d_losses.append(d_loss)
        g_losses.append(g_loss)

        if epoch % config.SAVE_INTERVAL == 0:
            print(f"Epoch: {epoch}, D Loss: {d_loss}, G Loss: {g_loss}")
            save_images(epoch, generator, z_dim, img_shape)
            generator.save(f'models/generator_epoch_{epoch}.keras')
            discriminator.save(f'models/discriminator_epoch_{epoch}.keras')

    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label="Discriminator Loss")
    plt.plot(g_losses, label="Generator Loss")
    plt.title("Training Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('training_losses.png')
    plt.show()

if __name__ == "__main__":
    train_gan(config.EPOCHS, config.BATCH_SIZE, config.Z_DIM, config.IMG_SHAPE)
