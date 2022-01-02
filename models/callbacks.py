import os
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt


class GANMonitor(Callback):
    def __init__(self, output_dir, ckpt_manager, num_img=16, latent_dim=100):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.output_dir = output_dir
        self.seed = tf.random.normal([num_img, latent_dim])
        self.ckpt_manager = ckpt_manager

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
        figpath = os.path.join(self.output_dir, "epoch_{:03d}.png".format(epoch+1))
        plt.savefig(figpath)
        plt.close(fig)

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = self.ckpt_manager.save()
            print("Saving checkpoint for epoch {} at {}".format(epoch+1, ckpt_save_path))