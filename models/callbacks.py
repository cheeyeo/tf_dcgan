import os
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt


class EpochCheckpoint(Callback):
    def __init__(self, output_dir, every=5, start_at=0, ckpt_obj=None):
        super(EpochCheckpoint, self).__init__()

        self.checkpoint_dir = output_dir
        self.every = every
        self.int_epoch = start_at
        self.checkpoint = ckpt_obj

    def on_epoch_end(self, epoch, logs=None):
        if (self.int_epoch + 1) % self.every == 0:
            checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
            self.checkpoint.save(file_prefix=checkpoint_prefix)

        self.int_epoch += 1


class GANMonitor(Callback):
    def __init__(self, output_dir, num_img=16, latent_dim=100, start_at=0):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.output_dir = output_dir
        self.seed = tf.random.normal([num_img, latent_dim])
        self.start_at = start_at

    def on_epoch_end(self, epoch, logs=None):
        generated_imgs = self.model.generator(self.seed)
        generated_imgs = (generated_imgs * 127.5) + 127.5
        generated_imgs.numpy()

        fig = plt.figure(figsize=(4, 4))
        for i in range(self.num_img):
            plt.subplot(4, 4, i+1)
            img = tf.keras.utils.array_to_img(generated_imgs[i])
            plt.imshow(img)
            plt.axis("off")
        figpath = os.path.join(self.output_dir, "epoch_{:03d}.png".format(self.start_at + 1))
        self.start_at += 1
        plt.savefig(figpath)
        plt.close(fig)