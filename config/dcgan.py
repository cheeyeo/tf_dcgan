import os


D_LR = 0.0001 # UPDATED: discriminator learning rate
G_LR = 0.0003 # UPDATED: generator learning rate

EPOCHS = 50
LATENT_DIM = 100
BATCH_SIZE = 32
HEIGHT = 64
WIDTH = 64
CHANNELS = 3

BASE_OUTPUT_DIR = os.path.join(os.getcwd(), "output")
MODEL_ARTIFACTS = os.path.join(BASE_OUTPUT_DIR, "model")
MODEL_CKPT = os.path.join(BASE_OUTPUT_DIR, "checkpoints")
PLOT_ARTIFACTS = os.path.join(BASE_OUTPUT_DIR, "plots")