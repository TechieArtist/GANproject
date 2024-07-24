from train import train_gan
import config

if __name__ == "__main__":
    train_gan(config.EPOCHS, config.BATCH_SIZE, config.Z_DIM, config.IMG_SHAPE)
