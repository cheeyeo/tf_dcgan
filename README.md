### GAN Training challenge

[DCGAN article]: https://www.pyimagesearch.com/2021/12/13/gan-training-challenges-dcgan-for-color-images/

[Clothing & Models]: https://www.kaggle.com/dqmonn/zalando-store-crawl


![Training visual of DCGAN](/assets/dcgan.gif)

Training a DCGAN with [Clothing & Models] dataset from Kaggle in an attempt to generate colour images using DCGANS.

The models is based on the following [DCGAN article]

The changes I made to the original codebase are:

* Added a checkpoint for model training using `tf.train.Checkpoint` and `tf.train.CheckpointManager`

* Added inference script for generating new images from sampling latent space for the generator

The DCGAN is trained for 50 epochs with batch size of 32. All the images are resized to 64x64x3.