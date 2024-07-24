from tensorflow.keras.callbacks import ModelCheckpoint

def create_checkpoint_callback(model_name):
    if not os.path.exists('models/'):
        os.makedirs('models/')
    checkpoint = ModelCheckpoint(f'models/{model_name}_epoch_{{epoch:02d}}.keras', save_freq='epoch')
    return checkpoint
