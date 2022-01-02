import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow logging (1)

import numpy as np
from imutils import paths
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # Suppress TensorFlow logging (2)

from tensorflow.data import AUTOTUNE
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

import config.dcgan as config
from models.dcgan import build_generator
from models.dcgan import build_discriminator
from models.dcgan import DCGAN
from models.callbacks import GANMonitor


if __name__ == "__main__":
    dataset = os.path.join(os.getcwd(), "dataset", "zalando", "zalando")

    # tf.keras.utils.image_dataset_from_directory returns a tf.data.Dataset obj
    # https://github.com/keras-team/keras/blob/v2.7.0/keras/preprocessing/image_dataset.py
    train_imgs = tf.keras.utils.image_dataset_from_directory(
        dataset,
        label_mode=None,
        image_size=(config.HEIGHT, config.WIDTH),
        batch_size=config.BATCH_SIZE
    )

    train_imgs = (train_imgs
        .map(lambda x: (x - 127.5) / 127.5)
    )

    print("[INFO] Building models...")
    generator = build_generator(config.LATENT_DIM, channels=config.CHANNELS)
    discriminator = build_discriminator(config.HEIGHT, config.WIDTH, config.CHANNELS)

    discriminator_optimizer = Adam(learning_rate=config.D_LR, beta_1=0.5)
    generator_optimizer = Adam(learning_rate=config.G_LR, beta_1=0.5)

    generator.summary()
    discriminator.summary()

    dcgan = DCGAN(discriminator=discriminator, generator=generator, latent_dim=config.LATENT_DIM)

    # set up checkpoints
    print("[INFO] Setting up checkpoints...")
    checkpoint_dir = os.path.join("output", "checkpoints")
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator
    )

    ckpt_manager = tf.train.CheckpointManager(
        checkpoint,
        checkpoint_dir, 
        max_to_keep=None
    )

    # if checkpoint exists, restore the latest checkpoint
    if ckpt_manager.latest_checkpoint:
        checkpoint.restore(ckpt_manager.latest_checkpoint)
        print("Restored Latest Checkpoint !!!")

    dcgan.compile(
        d_opt=discriminator_optimizer,
        g_opt=generator_optimizer,
        loss_fn=BinaryCrossentropy()
    )

    dcgan.fit(
        train_imgs,
        epochs=config.EPOCHS,
        callbacks=[GANMonitor(config.PLOT_ARTIFACTS, ckpt_manager, num_img=16, latent_dim=config.LATENT_DIM)]
    )

    dcgan.generator.save(config.MODEL_ARTIFACTS)