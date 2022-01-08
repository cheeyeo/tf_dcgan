### GAN Training challenge

[DCGAN article]: https://www.pyimagesearch.com/2021/12/13/gan-training-challenges-dcgan-for-color-images/

[Clothing & Models]: https://www.kaggle.com/dqmonn/zalando-store-crawl


![Training visual of DCGAN](/assets/dcgan.gif)

Training a DCGAN with [Clothing & Models] dataset from Kaggle in an attempt to generate colour images using DCGANS.

The models is based on the following: [DCGAN article]

The changes I made to the original codebase are:

* Added a custom callback for saving and loading model checkpoints using `tf.train.Checkpoint`

* Added inference / generation script for generating new images from sampling latent space for the generator

The DCGAN is trained for 50 epochs with batch size of 32. All the images are resized to 64x64x3 and with a latent dimension of 100.

There is also a variant of DCGAN called LSGAN which uses mean squared error to improve the GAN training stability and to create higher-res images but its not working for the current dataset.