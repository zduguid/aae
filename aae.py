# /usr/local/bin/python3
# 
# - implementation of Adversarial Autoencoder (AAE) to enable the capability
#   of making bathymetric predictions given sparse sonar readings from AUV 

from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import Lambda, GaussianNoise, BatchNormalization
from keras.layers import Activation, Embedding, ZeroPadding2D, MaxPooling2D
from keras.layers import LeakyReLU
import keras.backend as K

import math, sys, random, warnings
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as seaborn
from pathlib import Path
from simulator import Bathymetry
from utils import Animation, BoundingBox, FileFormatError


class AdversarialAutoencoder():
    """
    creates adversarial autoencoder that learns on multimodal bathymetry data
        + makes bathymetry predictions given denoising modality knockout
    """
    def __init__(self, animations=True):
        # define input dimensions and latent space dimension
        self.x_rows  = 80   # number of rows in input
        self.x_cols  = 80   # number of columns in input
        self.x_depth = 2    # two modalities being considered
        self.z_dim   = 10    # dimension of latent space, TBD
        self.y_dim   = 1    # dimension of the truth value 
        self.x_shape = (self.x_rows, self.x_cols, self.x_depth)
        self.gen_hidden_dim = 512
        self.dec_hidden_dim = 512
        self.dis_hidden_dim = 512
        self.animations     = animations

        # constants for activation and optimizer
        self.leaky = 0.2    # used for leaky rectified linear activation
        lr = 0.0002         # learning rate, parameter of Adam optimizer
        b1 = 0.5            # beta_1, parameter of Adam optimizer
        b2 = 0.999          # beta_2, parameter of Adam optimizer
        optimizer = Adam(lr, b1, b2)

        
        # build the networks involved in the adversarial autoencoder
        #   generator       (x -> z)
        #   decoder         (z -> x)
        #   autoencoder     (x -> x')
        #   discriminator   (z -> y)
        #       + discriminator trainable is set to False because generator
        #         and discriminator are trained in alternating phases
        self.generator     = self.build_generator()
        self.decoder       = self.build_decoder()
        self.autoencoder   = Model(self.generator.inputs, self.decoder(self.generator(self.generator.inputs)))
        self.discriminator = self.build_discriminator()
        self.discriminator.trainable = False

        # build the advarsarial autoencoder
        x      = Input(shape=self.x_shape)
        z      = self.generator(x)
        x_pred = self.decoder(z)
        y_fake = self.discriminator(z)
        self.aae = Model(x, [x_pred, y_fake])
        self.aae.compile(loss=['mse', 'binary_crossentropy'],
                         loss_weights=[0.999, 0.001],
                         optimizer=optimizer)

        # change discriminator to trainable and then compile 
        self.discriminator.trainable = True
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # print a summary of the models
        self.generator.summary()
        self.decoder.summary()
        self.discriminator.summary()
        self.aae.summary()


    def build_generator(self):
        """
        builds the generator neural network
            + a GAN's generator is the same as an Autoencoder's encoder
            + transforms input (x) to latent variable (z)
            + mu and log(sigma^2) are used to help stabilize learning
        :returns: the generator
        """
        # input
        x = Input(shape=self.x_shape, name='generator_x')

        # build the hidden layers
        h = Flatten()(x)
        h = Dense(self.gen_hidden_dim, name='generator_h1')(h)
        h = LeakyReLU(alpha=self.leaky)(h)
        h = Dense(self.gen_hidden_dim, name='generator_h2')(h)
        h = LeakyReLU(alpha=self.leaky)(h)
        mu = Dense(self.z_dim, name='generator_mu')(h)
        log_sigma_sq = Dense(self.z_dim, name='generator_log_sigma_sq')(h)

        # function to transforms mu and log(sigma^2) into z representation
        def get_z(args):
            mu, log_sigma_sq = args
            return mu + K.random_normal(K.shape(mu)) * K.exp(log_sigma_sq / 2)

        # get z representation from mu and log(sigma^2)
        z = Lambda(get_z)([mu, log_sigma_sq])

        return Model(x, z, name='generator')


    def build_decoder(self):
        """
        builds the decoder neural network
            + transforms latent variable (z) into reconstructed input (x')
            + hanging last dimension for Input's shape is for mini-batch size
        :returns: the decoder
        """
        # initialize the model
        model = Sequential()

        # build the hidden layers
        model.add(Dense(self.dec_hidden_dim, input_dim=self.z_dim, name='decoder_h1'))
        model.add(LeakyReLU(alpha=self.leaky))
        model.add(Dense(self.dec_hidden_dim, name='decoder_h2'))
        model.add(LeakyReLU(alpha=self.leaky))
        model.add(Dense(np.prod(self.x_shape), activation='tanh'))
        model.add(Reshape(self.x_shape))

        # get x from z using the model
        z = Input(shape=(self.z_dim,), name='decoder_z')
        x = model(z)

        return Model(z, x, name='decoder')


    def build_discriminator(self):
        """
        build the discriminator neural network
            + transforms latent variable (z) into truth value (y)
            + hanging last dimension for Input's shape is for mini-batch size
        :return: the discriminator
        """
        # initialize the model
        model = Sequential()

        # build the hidden layers
        model.add(Dense(self.dis_hidden_dim, input_dim=self.z_dim, name='discriminator_h1'))
        model.add(LeakyReLU(alpha=self.leaky))
        model.add(Dense(int(self.dis_hidden_dim/2), name='discriminator_h2'))
        model.add(LeakyReLU(alpha=self.leaky))
        model.add(Dense(self.y_dim, activation='sigmoid', name='discriminator_y'))

        # get y from z using the model
        z = Input(shape=(self.z_dim,))
        y = model(z)

        return Model(z, y, name='discriminator')


    def train(self, x, epochs, batch_size=32, sample_interval=50):
        """
        trains the AAE network by alternating between two phases:
            1) reconstruction phase: update generator and decoder to 
                minimize reconstruction error 
            2) regularization phase: update discriminator to distinguish true 
                samples from generated, update generator to fool discriminator
        :param x: the training data
        :param epochs: the number of epochs or passes through the data
        :param batch_size: the number of training examples in one batch
        :param sample_interval: determines when sample of model is taken
        """
        # TODO fix epoch formulation
        """
        We can divide the dataset of 2000 examples into batches of 500 
        then it will take 4 iterations to complete 1 epoch
        """
        # train for specified number of epochs
        for epoch in range(epochs):

            # get a batch of data points
            x_indices = np.random.randint(0, x.shape[0], batch_size)
            x_batch   = x[x_indices]

            # train the discriminator and then the generator
            dis_loss = self._train_discriminator(x_batch)
            gen_loss = self._train_generator(x_batch)
            
            if self.animations:
                d_loss = str(round(dis_loss[0], 3))
                d_acc  = str(round(dis_loss[1]*100, 3))
                g_loss = str(round(gen_loss[0], 3))
                g_mse  = str(round(gen_loss[1], 3))
                print("%d [D loss: %s, acc: %s%%] [G loss: %s, mse: %s]" % 
                      (epoch, d_loss, d_acc, g_loss, g_mse))

            if epoch % sample_interval == 0:
                pass


    def _train_generator(self, x_batch):
        """
        trains the generator on a batch of data
        """
        batch_size = len(x_batch)

        # get real y values to train generator 
        y_real = np.ones((batch_size, 1))
        
        # return the loss of the generator
        return self.aae.train_on_batch(x_batch, [x_batch, y_real])


    def _train_discriminator(self, x_batch):
        """
        trains the discriminator on a batch of data
        """
        batch_size = len(x_batch)
        
        # get real and fake y values to train discriminator 
        y_real = np.ones((batch_size, 1))
        y_fake = np.zeros((batch_size, 1))

        # get real and fake z values to train discriminator
        z_real = np.random.normal(size=(batch_size, self.z_dim))
        z_fake = self.generator.predict(x_batch)
        
        # train discriminator on real and fake data
        dis_loss_real = self.discriminator.train_on_batch(z_real, y_real)
        dis_loss_fake = self.discriminator.train_on_batch(z_fake, y_fake)

        # return the average of the two losses
        return 0.5 * np.add(dis_loss_fake, dis_loss_fake)


    def generate_samples(self, epoch):
        """
        TODO write specification
        """
        # TODO implement this function
        pass


    def save_model(self):
        """
        TODO write specification
        """
        # TODO implement this function
        pass


if __name__ == '__main__':
    # raw data set
    #   + download link: (http://www.soest.hawaii.edu/pibhmc/cms/)
    raw_file  = 'data/kohala/kohala_synth_5m.asc'
    raw_bb    = BoundingBox(w_lim = -156.31, 
                            e_lim = -155.67, 
                            n_lim =   20.54, 
                            s_lim =   19.64)

    # Falkor data set where engineering cruise took place in Hawaii
    #   + more information about Falkor: (https://schmidtocean.org/rv-falkor/)
    falkor_file = 'data/falkor/falkor_5m.npy'
    falkor_bb = BoundingBox(w_lim = -156.03, 
                            e_lim = -155.82, 
                            n_lim =   20.01, 
                            s_lim =   19.84)

    # ignore specific warning that is thrown due to the nature of AAE structure
    warnings.filterwarnings(action='once', message='Discrepancy between trainable weights and collected trainable')
    aae = AdversarialAutoencoder()

    # load the data and begin training
    data = np.load('data/simulated/sonar_medium_80.npy')
    aae.train(x=data, epochs=1000, batch_size=32, sample_interval=10)
