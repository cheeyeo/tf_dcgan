import argparse
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
from models.callbacks import EpochCheckpoint


if __name__ == "__main__":
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--start", type=int, default=0, help="Resume training from specific epoch checkpoint")
    # args = vars(ap.parse_args())
    # print(args)


    print("[INFO] Preparing dataset...")
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

    d_opt = Adam(learning_rate=config.D_LR, beta_1=0.5)
    g_opt = Adam(learning_rate=config.G_LR, beta_1=0.5)

    generator.summary()
    discriminator.summary()

    dcgan = DCGAN(discriminator=discriminator, generator=generator, latent_dim=config.LATENT_DIM)
   
    dcgan.compile(
        d_opt=d_opt,
        g_opt=g_opt,
        loss_fn=BinaryCrossentropy()
    )

    print("[INFO] Checking for latest checkpoint ...")
    ckpt_dir = os.path.join("output", "checkpoints")
    # when to start checkpoint; will be 0 when first training
    start_at = 0

    # define the objects we want to persist
    ckpt_obj = tf.train.Checkpoint(
            d_opt=d_opt,
            g_opt=g_opt,
            generator=generator,
            discriminator=discriminator
    )

    latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)

    if latest_ckpt is not None:
        print("[INFO] Resuming from ckpt: {}".format(latest_ckpt))
        ckpt_obj.restore(latest_ckpt).assert_existing_objects_matched().expect_partial()
        latest_ckpt_idx = latest_ckpt.split(os.path.sep)[-1].split("-")[-1]
        start_at = int(latest_ckpt_idx)
        print(f"[INFO] Resuming ckpt at {start_at}")

    print("[INFO] Setting up callbacks...")
    ckpt_callback = EpochCheckpoint(ckpt_dir, every=1, start_at=start_at, ckpt_obj=ckpt_obj)
    gan_monitor = GANMonitor(config.PLOT_ARTIFACTS, num_img=16, latent_dim=config.LATENT_DIM, start_at=start_at)

    dcgan.fit(
        train_imgs,
        epochs=config.EPOCHS,
        callbacks=[
            ckpt_callback,
            gan_monitor
        ]
    )

    dcgan.generator.save(config.MODEL_ARTIFACTS)