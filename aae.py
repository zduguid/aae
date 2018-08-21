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

import math, sys, random, warnings, os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns
from pathlib import Path
from simulator import Bathymetry
from utils import Animation, BoundingBox, FileFormatError


class AdversarialAutoencoder():
    """
    creates adversarial autoencoder that learns on multimodal bathymetry data
        + makes bathymetry predictions given sparse sonar readings 
    """
    def __init__(self, animations=True):
        # define input dimensions and latent space dimension
        self.x_rows  = 50   # number of rows in input
        self.x_cols  = 50   # number of columns in input
        self.z_dim   = 100  # dimension of latent space
        self.y_dim   = 1    # dimension of the truth value 
        self.x_shape = (self.x_rows, self.x_cols)
        self.gen_hidden_dim = 512
        self.dec_hidden_dim = 512
        self.dis_hidden_dim = 512
        self.animations     = animations

        # constants for activation and optimizer
        self.leaky = 0.2    # used for leaky rectified linear activation
        lr = 0.0001         # learning rate (default = 0.001)
        b1 = 0.5            # beta_1 (default = 0.9)
        b2 = 0.999          # beta_2 (default = 0.999)
        optimizer = Adam(lr, b1, b2)

        
        # build the networks involved in the adversarial autoencoder
        #   encoder         (x -> z)
        #   decoder         (z -> x)
        #   autoencoder     (x -> x')
        #   discriminator   (z -> y)
        #       + discriminator trainable is set to False because encoder
        #         and discriminator are trained in alternating phases
        self.encoder       = self.build_encoder()
        self.decoder       = self.build_decoder()
        self.autoencoder   = Model(self.encoder.inputs, 
                                   self.decoder(self.encoder(self.encoder.inputs)))
        self.discriminator = self.build_discriminator()
        self.discriminator.trainable = False

        # build the advarsarial autoencoder
        x      = Input(shape=self.x_shape)
        z      = self.encoder(x)
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
        self.encoder.summary()
        self.decoder.summary()
        self.autoencoder.summary()
        self.discriminator.summary()
        self.aae.summary()


    def build_encoder(self):
        """
        builds the encoder neural network
            + a GAN's generator is the same as an Autoencoder's encoder
            + transforms input (x) to latent variable (z)
            + log(sigma^2) is used instead of sigma to stabilize learning
        :returns: the encoder
        """
        # input
        x = Input(shape=self.x_shape, name='encoder_x')

        # build the hidden layers
        h = Flatten()(x)
        h = Dense(self.gen_hidden_dim, name='encoder_h1')(h)
        h = LeakyReLU(alpha=self.leaky)(h)
        h = Dense(self.gen_hidden_dim, name='encoder_h2')(h)
        h = LeakyReLU(alpha=self.leaky)(h)
        mu = Dense(self.z_dim, name='encoder_mu')(h)
        log_sigma_sq = Dense(self.z_dim, name='encoder_log_sigma_sq')(h)

        # function to transforms mu and log(sigma^2) into z representation
        def get_z(args):
            mu, log_sigma_sq = args
            return mu + K.random_normal(K.shape(mu)) * K.exp(log_sigma_sq / 2)

        # get z representation from mu and log(sigma^2)
        z = Lambda(get_z)([mu, log_sigma_sq])

        return Model(x, z, name='encoder')


    def build_decoder(self):
        """
        builds the decoder neural network
            + transforms latent variable (z) into reconstructed input (x')
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
        :returns: the discriminator
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


    def train(self, data_bath, data_sonar, epochs, batch_size=32, sample_interval=100):
        """
        trains the AAE network by alternating between two phases:
            1) reconstruction phase: update encoder and decoder to 
                minimize reconstruction error 
            2) regularization phase: update discriminator to distinguish true 
                samples from generated, update encoder to fool discriminator
        :param data_bath:  the sampled bathymetry patch training data
        :param data_sonar: the simulated sonar readings training data
        :param epochs: the number of passes through the data
        :param batch_size: the number of training examples in a training batch
        :param sample_interval: frequency of generating a test sample
        """
        if self.animations: print('>> training network')

        # assemble all training data into one array using defensive copying
        #   + train on the bathymetry twice as often as the sonar
        data = np.vstack([np.copy(data_bath), 
                          np.copy(data_bath),
                          np.copy(data_sonar)])

        # pass through entire data set a specified number of times
        for epoch in range(epochs):

            # randomize order that data is presented for a given epoch
            epoch_i = random.sample(range(len(data)), len(data))

            # train the network in batches of size batch_size
            for j in range(0, len(epoch_i), batch_size):

                # get the batch of data points for one training update
                x_batch = np.copy(data[epoch_i[j : j+batch_size]])

                # train the discriminator and then the encoder
                dis_loss = self._train_discriminator(x_batch)
                gen_loss = self._train_encoder(x_batch)
                
                # print out loss statistics for the training iteration
                if self.animations:
                    d_loss = str(round(dis_loss[0], 3))
                    d_acc  = str(round(dis_loss[1]*100, 3))
                    g_loss = str(round(gen_loss[0], 3))
                    g_mse  = str(round(gen_loss[1], 3))
                    print("%d [D loss: %s, acc: %s%%] [G loss: %s, mse: %s]" % 
                          (epoch, d_loss, d_acc, g_loss, g_mse))

            # generate example predictions at specified interval
            if (epoch+1) % sample_interval == 0:
                
                # get sampled data points by sampling from p(z)
                self._get_samples(epoch + 1)

                # get predicted bathymetry values using sonar alone
                self._get_predictions(epoch + 1, data_bath, data_sonar)


    def _train_encoder(self, x_batch):
        """
        trains the encoder on a batch of data
        """
        batch_size = len(x_batch)

        # adversarial ground truth 
        y_real = np.ones((batch_size, 1))
        
        # return the loss of the encoder
        return self.aae.train_on_batch(x_batch, [x_batch, y_real])


    def _train_discriminator(self, x_batch):
        """
        trains the discriminator on a batch of data
        """
        batch_size = len(x_batch)
        
        # adversarial ground truths 
        y_real = np.ones((batch_size, 1))
        y_fake = np.zeros((batch_size, 1))

        # get real and fake z values to train discriminator
        z_real = np.random.normal(size=(batch_size, self.z_dim))
        z_fake = self.encoder.predict(x_batch)
        
        # train discriminator on real and fake data
        dis_loss_real = self.discriminator.train_on_batch(z_real, y_real)
        dis_loss_fake = self.discriminator.train_on_batch(z_fake, y_fake)

        # return the average of the two losses
        return 0.5 * np.add(dis_loss_fake, dis_loss_fake)


    def _get_samples(self, epoch):
        """
        generates sample data points by sampling z values from p(z)
        :param epochs: the epoch at which the samples are generated 
        """
        # plotting parameters
        ncols = 4
        figsize = (14, 4)
        font_large = 25
        font_medium = 15
        sns.set_style('darkgrid')

        # initialize the plotting objects
        fig, ax = plt.subplots(figsize=figsize, ncols=ncols, nrows=1)
        plt.subplots_adjust(left    =  0.1,     # left side location
                            bottom  =  0.1,     # bottom side location
                            right   =  0.9,     # right side location
                            top     =  0.9,     # top side location
                            wspace  =  0.6,     # horizontal gap
                            hspace  =  0.05)    # vertical gap 

        # generate each column of the plot
        for i in range(ncols):
            # sample from p(z) and generate x_pred with the decoder
            z_point    = np.random.normal(size=(1, self.z_dim))
            data_point = self.decoder.predict(z_point)[0]

            # extract min and max for color scaling
            vmin = np.min(data_point)
            vmax = np.max(data_point)

            # plot the corresponding patch of bathymetry
            sns.heatmap(data_point, square=True, cmap='jet', 
                        vmin=vmin, vmax=vmax, ax=ax[i],
                        xticklabels=False, yticklabels=False,
                        cbar=False)

        fig.suptitle('Sampling Bathymetry Patches \n AAE Network (Epoch ' + str(epoch) + ')', fontsize=font_large)
        ax[0].set(ylabel='Sampled \n Bathymetry Patch')
        ax[0].yaxis.label.set_size(font_medium)
        for i in range(ncols):
            ax[i].set(xlabel='Example '+str(i+1))
            ax[i].xaxis.label.set_size(font_medium)
        plt.savefig('data/plots/sampled_epoch' + str(epoch) + '.png')
        plt.close()


    def _get_predictions(self, epoch, data_bath, data_sonar):
        """
        generates predictions of bathymetry patches using given data
        :param epochs: the epoch at which the predictions are generated
        :param data: the data to be randomly sampled to make predictions 
        """
        # plotting parameters
        ncols = min(4, len(data_sonar))
        figsize = (14, 9)
        font_large = 25
        font_medium = 15
        sns.set_style('darkgrid')

        # randomly sample points in the data set
        sample_indices = random.sample(range(len(data_sonar)), ncols)
        sonar_points   = data_sonar[sample_indices]
        bath_points    = data_bath[sample_indices]

        # mask used for plotting simulated sonar readings
        def mask(x):
            return x == 0

        # initialize the plotting objects
        fig, ax = plt.subplots(figsize=figsize, ncols=ncols, nrows=3)
        plt.subplots_adjust(left    =  0.1,     # left side location
                            bottom  =  0.1,     # bottom side location
                            right   =  0.9,     # right side location
                            top     =  0.9,     # top side location
                            wspace  =  0.6,     # horizontal gap
                            hspace  =  0.05)    # vertical gap 

        # generate each column of the plot
        for i in range(ncols):
            sonar_point = sonar_points[i]
            bath_point  = bath_points[i]

            # generate the input to the autoencoder
            x_input = np.expand_dims(sonar_point, axis=0)

            # retrieve the autoencoder prediction
            x_output = self.autoencoder.predict(x_input)[0]

            # extract min and max for color scaling
            vmin = np.min(bath_point)
            vmax = np.max(bath_point)

            # plot the ground truth bathymetry data
            sns.heatmap(bath_point, square=True, cmap='jet', 
                        vmin=vmin, vmax=vmax, ax=ax[0][i],
                        xticklabels=False, yticklabels=False,
                        cbar=False)

            # plot the simulated sonar data
            sns.heatmap(sonar_point, square=True, cmap='jet', 
                        vmin=vmin, vmax=vmax, ax=ax[1][i],
                        xticklabels=False, yticklabels=False,
                        mask=mask(sonar_point),
                        cbar=False)

            # plot the predicted bathymetry data
            sns.heatmap(x_output, square=True, cmap='jet', 
                        vmin=vmin, vmax=vmax, ax=ax[2][i],
                        xticklabels=False, yticklabels=False,
                        cbar=False)

        # handle labeling of subplots
        fig.suptitle('Predicting Bathymetry Patches \n AAE Network (Epoch ' + str(epoch) + ')', fontsize=font_large)
        ax[0][0].set(ylabel='Ground Truth \n Bathymetry Patch')
        ax[0][0].yaxis.label.set_size(font_medium)
        ax[1][0].set(ylabel='Simulated Sonar \n Readings')
        ax[1][0].yaxis.label.set_size(font_medium)
        ax[2][0].set(ylabel='Predicted \n Bathymetry Patch')
        ax[2][0].yaxis.label.set_size(font_medium)
        for i in range(ncols):
            ax[2][i].set(xlabel='Sample '+str(i+1))
            ax[2][i].xaxis.label.set_size(font_medium)
        plt.savefig('data/plots/predicted_epoch' + str(epoch) + '.png')
        plt.close()


    def save_model(self):
        """
        saves all neural networks involved in the adversarial autoencoder
        """
        # helper function that saves model with name
        def save(model, name):
            # generate file paths for writing files
            path_model   = 'data/models/%s.json' % name
            path_weights = 'data/models/%s_weights.hdf5' % name
            options = {'file_arch'   : path_model,
                       'file_weight' : path_weights}

            # write to files accordingly
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        # save each model in the aae
        save(self.encoder,       'aae_encoder')
        save(self.decoder,       'aae_decoder')
        save(self.autoencoder,   'aae_autoencoder')
        save(self.discriminator, 'aae_discriminator')


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

    # load bathymetry file and training data
    bath = Bathymetry.load_file(falkor_file, falkor_bb)
    data_bath  = np.load('data/simulated/data_bath_n5000_50x50.npy')
    data_sonar = np.load('data/simulated/data_sonar_n5000_50x50.npy')

    # construct the adversarial autoencoder
    warnings.filterwarnings(action='once', message='Discrepancy between trainable weights and collected trainable')
    aae = AdversarialAutoencoder()

    # train the adversarial autoencoder with specified parameters and data
    aae.train(data_bath=data_bath, 
              data_sonar=data_sonar, 
              epochs=1000, 
              batch_size=32, 
              sample_interval=100)

    # save the model
    aae.save_model()