import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Reshape, Dense, ReLU, Conv2DTranspose, Conv2D, BatchNormalization, LeakyReLU, Flatten, Dropout, Activation


WEIGHT_INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

def build_generator(input_dim=1024, channels=3):
    """
    generated images of shape 112x112x3
    """
    model = Sequential(name="generator")

    model.add(Dense(7 * 7 * 512, input_dim=input_dim))
    # model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Reshape((7, 7, 512)))

    # upsample to 14x14: apply a transposed CONV => BN => RELU
    model.add(Conv2DTranspose(256, (4, 4), strides=(2, 2),padding="same", kernel_initializer=WEIGHT_INIT))
    # model.add(BatchNormalization())
    model.add((ReLU()))

    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2),padding="same", kernel_initializer=WEIGHT_INIT))
    # model.add(BatchNormalization())
    model.add((ReLU()))

    model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2),padding="same", kernel_initializer=WEIGHT_INIT))
    # model.add(BatchNormalization())
    model.add((ReLU()))

    model.add(Conv2DTranspose(3, (4, 4), strides=(2, 2),padding="same", kernel_initializer=WEIGHT_INIT))
    model.add(Activation("tanh"))

    return model


def build_discriminator(height, width, depth, alpha=0.02):
    model = Sequential(name='discriminator')
    input_shape = (height, width, depth)

    # 1. first set of CONV => BN => leaky ReLU layers
    model.add(Conv2D(64, (4, 4), padding="same", strides=(2, 2),
        input_shape=input_shape))
    model.add(LeakyReLU(alpha=alpha))

    # 2. second set of CONV => BN => leacy ReLU layers
    model.add(Conv2D(128, (4, 4), padding="same", strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=alpha))

    # 3. third set of CONV => BN => leacy ReLU layers
    model.add(Conv2D(256, (4, 4), padding="same", strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=alpha))

    # 4. fourth set of CONV => BN => leacy ReLU layers
    model.add(Conv2D(512, (4, 4), padding="same", strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=alpha))

    model.add(Flatten())
    # model.add(Dropout(0.3))

    model.add(Dense(1, activation="linear"))
    return model


class LSGAN(Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(LSGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")

    def compile(self, d_opt, g_opt, loss_fn):
        super(LSGAN, self).compile()
        self.d_opt = d_opt
        self.g_opt = g_opt
        self.loss_fn = loss_fn

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Train discriminator with real imgs labeled as 1 and fake imgs labeled with 0
        with tf.GradientTape() as tape:
            pred_real = self.discriminator(real_images, training=True)
            real_labels = tf.ones((batch_size, 1))
            real_labels += 0.05 * tf.random.uniform(tf.shape(real_labels))

            d_loss_real = self.loss_fn(real_labels, pred_real)

            # compute loss on fake images
            fake_imgs = self.generator(noise)
            pred_fake = self.discriminator(fake_imgs, training=True)
            fake_labels = tf.zeros((batch_size, 1))
            d_loss_fake = self.loss_fn(fake_labels, pred_fake)

            d_loss = (d_loss_real + d_loss_fake) / 2

        # compute discriminator gradients
        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)

        # update discriminator weights
        self.d_opt.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        # train generator but dont update weight of discriminator
        misleading_labels = tf.ones((batch_size, 1))

        with tf.GradientTape() as tape:
            fake_imgs = self.generator(noise, training=True)
            pred_fake = self.discriminator(fake_imgs, training=True)
            g_loss = self.loss_fn(misleading_labels, pred_fake)

        # compute generator gradients
        grads = tape.gradient(g_loss, self.generator.trainable_variables)

        # update generator weights
        self.g_opt.apply_gradients(zip(grads, self.generator.trainable_variables))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {"d_loss": self.d_loss_metric.result(), "g_loss": self.g_loss_metric.result()}


if __name__ == "__main__":
    g = build_generator(1024, 3)
    g.summary()

    # d = build_discriminator(112, 112, 3)
    # d.summary()