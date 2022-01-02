# Use pretrained generator model to generate new images
import argparse

import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to trained generator model")
    ap.add_argument("--output", default="test.png", help="Output filename to save generated image")
    args = vars(ap.parse_args())

    model_path = args["model"]

    generator = load_model(model_path)
    generator.summary()

    seed = tf.random.normal([1, 100])
    print(seed.shape)

    generated_img = generator(seed)
    print(generated_img.shape)
    generated_img = (generated_img[0] * 127.5) + 127.5
    img = tf.keras.utils.array_to_img(generated_img)
    plt.figure()
    plt.imshow(img)
    plt.savefig(args["output"])