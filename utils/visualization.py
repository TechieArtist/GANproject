import matplotlib.pyplot as plt
import numpy as np
import os

def save_images(epoch, generator, z_dim, img_shape, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.normal(0, 1, (examples, z_dim))
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, *img_shape)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    if not os.path.exists('images/'):
        os.makedirs('images/')
    plt.savefig(f'images/gan_generated_image_epoch_{epoch}.png')
