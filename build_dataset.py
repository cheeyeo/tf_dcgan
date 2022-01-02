import os

import numpy as np
from imutils import paths
import tensorflow as tf
from tensorflow.data import AUTOTUNE
import matplotlib.pyplot as plt

import config.dcgan as config


if __name__ == "__main__":
    dataset = os.path.join(os.getcwd(), "dataset", "zalando", "zalando")

    # https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
    train_imgs = tf.keras.utils.image_dataset_from_directory(
        dataset,
        label_mode=None,
        image_size=(config.HEIGHT, config.WIDTH),
        batch_size=config.BATCH_SIZE
    )

    train_imgs = (train_imgs
        .map(lambda x: (x - 127.5) / 127.5)
    )

    img_batch = next(iter(train_imgs))

    fig = plt.figure(figsize=(8, 8))

    for i in range(0, config.BATCH_SIZE):
        img = img_batch[i].numpy()
        img = (img * 127.5) + 127.5
        img = img.astype("uint8")
        ax = plt.subplot(8, 4, i+1)
        plt.imshow(img)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("visualize.png")