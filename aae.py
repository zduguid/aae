# /usr/local/bin/python3
# 
# - implementation of Adversarial Autoencoder (AAE) to enable the capability
#   of making bathymetric predictions given sparse sonar readings from AUV 

from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import merge, GaussianNoise, BatchNormalization
from keras.layers import Activation, Embedding, ZeroPadding2D, MaxPooling2D
from keras.layers import LeakyReLU
import keras.backend as K

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class AdversarialAutoencoder():
    """
    TODO write specification
    """
    def __init__(self):
        # define input dimensions and latent space dimension
        self.x_rows  = 100      # number of rows in input, TBD
        self.x_cols  = 100      # number of columns in input, TBD
        self.x_depth = 2        # two modalities being considered
        self.z_dim   = 10       # dimension of latent space, TBD
        self.x_shape = (self.x_rows, self.x_cols, self.x_depth)

        # other constants
        self.leakiness = 0.2    # used for leaky rectified linear activation
        self.encoder_hidden_dim   = 512
        self.generator_hidden_dim = 512

        # define the optimizer to be used during learning
        lr = 0.001              # learning rate
        b1 = 0.9                # beta_1, parameter of Adam optimizer
        b2 = 0.999              # beta_2, parameter of Adam optimizer
        optimizer = Adam(lr, b1, b2)

        # encoder (x -> z)
        self.encoder = self.build_encoder()
        # generator (z -> x)
        self.generator = self.build_generator()
        # autoencoder (x -> x')
        self.autoencoder = Model(self.encoder.inputs, self.generator(self.encoder(self.encoder.inputs)))
        # discriminator (z -> y)
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
        # discriminator and generator are trained in alternating fashion
        self.discriminator.trainable = False

        # build AAE
        x      = Input(shape=self.x_shape)
        z      = self.encoder(x)
        x_pred = self.generator(z)
        y_fake = self.discriminator(z)
        self.aae = Model(x, [x_pred, y_fake])
        self.aae.compile(loss = ['mse', 'binary_crossentropy'],
                         loss_weights = [0.999, 0.001],
                         optimizer = optimizer)


    def build_encoder(self):
        """
        builds the encoder NN
        :returns: encoder NN
        """
        # input
        x = Input(shape=self.x_shape, name="encoder_x")

        # build the hidden layers
        h = Flatten()(x)
        h = Dense(self.encoder_hidden_dim, name="encoder_h1")(h)
        h = LeakyReLU(alpha=self.leakiness)(h)
        h = Dense(self.encoder_hidden_dim, name="encoder_h2")(h)
        h = LeakyReLU(alpha=self.leakiness)(h)

        # encoder generates mu and sigma of latent distribution z
        #   + log(sigma^2) is used to provide stability during learning
        mu = Dense(self.z_dim, name="encoder_mu")(h)
        log_sigma_sq = Dense(self.z_dim, name="encoder_log_sigma_sq")(h)
        z = merge([mu, log_sigma_sq],
                  mode=lambda p: p[0] + K.random_normal(K.shape(p[0])) * K.exp(p[1] / 2),
                  output_shape=lambda p: p[0])

        return Model(x, z, name="encoder")


    def build_generator(self):
        """
        builds the generator NN
        :returns: generator NN
        """
        model = Sequential(name="generator")
        model.add(Dense(self.generator_hidden_dim, input_dim=self.z_dim, name="generator_h1"))
        model.add(LeakyReLU(alpha=self.leakiness))
        model.add(Dense(self.generator_hidden_dim, name="generator_h2"))
        model.add(LeakyReLU(alpha=self.leakiness))
        model.add(Dense(np.prod(self.x_shape)))
        model.add(Activation("sigmoid"))
        model.add(Reshape(self.x_shape))

        z = Input(shape=(self.z_dim,), name="generator_z")
        x = model(z)

        return Model(z, x)



    def build_discriminator(self):
        """
        build the discriminator NN
        :return: discriminator NN
        """
        # TODO network structure 
        model = Sequential()

        model.add(Dense(512, input_dim=self.z_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation="sigmoid"))
        model.summary()

        encoded_repr = Input(shape=(self.latent_dim, ))
        validity = model(encoded_repr)

        return Model(encoded_repr, validity)



    def train(self, epochs, batch_size, sample_interval):
        """
        TODO write specification
        """
        # TODO implement this function
        pass


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


if __name__ == "__main__":
    aae = AdversarialAutoencoder()

