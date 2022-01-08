import os

D_LR = 0.0001 # UPDATED: discriminator learning rate
G_LR = 0.0003 # UPDATED: generator learning rate

EPOCHS = 50
# BATCH_SIZE = 64
# LATENT_DIM = 128

BATCH_SIZE = 32
LATENT_DIM = 100
HEIGHT = 112
WIDTH = 112
CHANNELS = 3


OUTPUT_BASE = os.path.join(os.getcwd(), "output")
CKPT_DIR = os.path.join(OUTPUT_BASE, "lsgan_checkpoints")
MODEL_ARTIFACTS = os.path.join(OUTPUT_BASE, "lsgan_model")
PLOT_ARTIFACTS = os.path.join(OUTPUT_BASE, "lsgan_plots")