from tensorflow.keras.models import Sequential

def build_gan(generator, discriminator):
    model = Sequential([generator, discriminator])
    return model
