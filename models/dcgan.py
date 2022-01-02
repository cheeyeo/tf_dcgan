import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Reshape, Dense, ReLU, Conv2DTranspose, Conv2D, BatchNormalization, LeakyReLU, Flatten, Dropout


WEIGHT_INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

def build_generator(input_dim, channels=3):
    model = Sequential(name='generator')

    # prepare for reshape: FC => BN => RN layers, note: input shape defined in the 1st Dense layer  
    model.add(Dense(8 * 8 * 512, input_dim=input_dim))
    model.add(ReLU())

    # 1D => 3D: reshape the output of the previous layer 
    model.add(Reshape((8, 8, 512)))

    # upsample to 16x16: apply a transposed CONV => BN => RELU
    model.add(Conv2DTranspose(256, (4, 4), strides=(2, 2),padding="same", kernel_initializer=WEIGHT_INIT))
    model.add((ReLU()))

    # upsample to 32x32: apply a transposed CONV => BN => RELU
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2),padding="same", kernel_initializer=WEIGHT_INIT))
    model.add((ReLU()))

    # upsample to 64x64: apply a transposed CONV => BN => RELU
    model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2),padding="same", kernel_initializer=WEIGHT_INIT))
    model.add((ReLU()))

    # final layer: Conv2D with tanh activation
    model.add(Conv2D(channels, (4, 4), padding="same", activation="tanh"))

    return model


def build_discriminator(height, width, depth, alpha=0.2):
    # create a Keras Sequential model
    model = Sequential(name='discriminator')
    input_shape = (height, width, depth)

    # 1. first set of CONV => BN => leaky ReLU layers
    model.add(Conv2D(64, (4, 4), padding="same", strides=(2, 2),
        input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=alpha))

    # 2. second set of CONV => BN => leacy ReLU layers
    model.add(Conv2D(128, (4, 4), padding="same", strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=alpha))

    # 3. third set of CONV => BN => leacy ReLU layers
    model.add(Conv2D(128, (4, 4), padding="same", strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=alpha))

    # flatten and apply dropout
    model.add(Flatten())
    model.add(Dropout(0.3))

    # sigmoid in the last layer outputting a single value for binary classification
    model.add(Dense(1, activation="sigmoid"))

    # return the discriminator model
    return model


class DCGAN(Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(DCGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")

    def compile(self, d_opt, g_opt, loss_fn):
        super(DCGAN, self).compile()
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
    m = build_generator(100, 3)
    m.summary()

    m = build_discriminator(64, 64, 3)
    m.summary()