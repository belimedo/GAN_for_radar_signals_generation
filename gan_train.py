# -*- coding: utf-8 -*-
"""
Created on Sat Apr  24 12:58:40 2021

Final cGAN training code

@author: milan.medic
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape,Embedding,Concatenate
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.models import load_model
from datetime import datetime 

# import util functions from util file
from gan_util import save_to_txt, summarize_cperformance, print_model_summary, save_cplot, print_gan_performance, plot_creal_data 
from gan_util import  load_training_data, generate_clatent_points, generate_cmeasured_samples, generate_cfake_samples

# This is path to the pickled data. This data is randomised and reduced to interval [-1,1]
DATA_PATH   = "two_class_data.pickle"
MODELS_DIR  = "models"
NXT_PATH    = "/home/research/milan_medic/trenirano_fakultet"
#NXT_PATH    = 'C:\\Users\\milan.medic\\Documents\\GAN'
ROOT_PATH    = os.path.join(NXT_PATH,MODELS_DIR)

if not os.path.exists(MODELS_DIR):
	os.mkdir(MODELS_DIR)

# Make sure to create root dir folder
if not os.path.exists(ROOT_PATH):
    os.mkdir(ROOT_PATH)

###################
# CGAN MODEL CODE #
###################

# This method is a safe guard that resets model if loss goes to 0
def reset_cmodels(epoch_num, reset_num, dir_name, strategy = None):
    
    dis_name = os.path.join(dir_name,'generators')
    dis_name = os.path.join(dis_name,f"discriminator_model_{epoch_num}.h5")
    gen_name = os.path.join(dir_name,'generators')
    gen_name = os.path.join(gen_name,f"generator_model_{epoch_num}.h5")
   
    if not os.path.exists(dis_name):
        print("Ne postoji!")
        return None,None,None
    
    if strategy != None:
        with strategy.scope():
            d_model = load_model(dis_name)
            d_model.summary()
            g_model = load_model(gen_name)
            g_model.summary()
            gan_model = define_cmodel_5_gan(g_model, d_model, optim = Adam(lr=0.0002/(reset_num+1), beta_1=0.5), strategy=strategy)
            gan_model.summary()
    else:
        d_model = load_model(dis_name)
        d_model.summary()
        g_model = load_model(gen_name)
        g_model.summary()
        gan_model = define_cmodel_5_gan(g_model, d_model, optim = Adam(lr=0.0002/(reset_num+1), beta_1=0.5), strategy=strategy)
        gan_model.summary()
        
    return d_model, g_model, gan_model


def define_cmodel_5_discriminator(in_shape=(128,300,1), n_classes=2, strategy = None):
    if strategy != None:
        with strategy.scope():
            # label input
            in_label = Input(shape=(1,))
            # embedding for categorical input
            lab_input = Embedding(n_classes, 75)(in_label)
            # scale up to image dimensions with linear activation
            n_nodes = in_shape[0] * in_shape[1]
            lab_input = Dense(n_nodes)(lab_input)
            # reshape to additional channel
            lab_input = Reshape((in_shape[0], in_shape[1], 1))(lab_input)
            # image input
            in_image = Input(shape=in_shape)
            # concat label as a channel
            merge = Concatenate()([in_image, lab_input])
            # perform one 2D convolution
            model = Conv2D(64, (3,3), padding='same')(merge)
            model = LeakyReLU(alpha=0.2)(model)
            # downsample to 64x150
            model = Conv2D(128, (3,3), strides=(2,2), padding='same')(model)
            model = LeakyReLU(alpha=0.2)(model)
            # downsample to 32x75
            model = Conv2D(128, (3,3), strides=(2,2), padding='same')(model)
            model = LeakyReLU(alpha=0.2)(model)
            # downsample to 16x25
            model = Conv2D(256, (3,3), strides=(2,3), padding='same')(model)
            model = LeakyReLU(alpha=0.2)(model)
            # downsample to 4x5
            model = Conv2D(256, (3,3), strides=(4,5), padding='same')(model)
            model = LeakyReLU(alpha=0.2)(model)
            # flatten feature maps
            model = Flatten()(model)
            # dropout
            model = Dropout(0.4)(model)
            # output
            out_layer = Dense(1, activation='sigmoid')(model)
            # define model
            model = Model([in_image, in_label], out_layer)
            # compile model
            opt = Adam(lr=0.0002, beta_1=0.5)
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
            return model
    else:
        # label input
        in_label = Input(shape=(1,))
        # embedding for categorical input
        lab_input = Embedding(n_classes, 75)(in_label)
        # scale up to image dimensions with linear activation
        n_nodes = in_shape[0] * in_shape[1]
        lab_input = Dense(n_nodes)(lab_input)
        # reshape to additional channel
        lab_input = Reshape((in_shape[0], in_shape[1], 1))(lab_input)
        # image input
        in_image = Input(shape=in_shape)
        # concat label as a channel
        merge = Concatenate()([in_image, lab_input])
        # perform one 2D convolution
        model = Conv2D(64, (3,3), padding='same')(merge)
        model = LeakyReLU(alpha=0.2)(model)
        # downsample to 64x150
        model = Conv2D(128, (3,3), strides=(2,2), padding='same')(model)
        model = LeakyReLU(alpha=0.2)(model)
        # downsample to 32x75
        model = Conv2D(128, (3,3), strides=(2,2), padding='same')(model)
        model = LeakyReLU(alpha=0.2)(model)
        # downsample to 16x25
        model = Conv2D(256, (3,3), strides=(2,3), padding='same')(model)
        model = LeakyReLU(alpha=0.2)(model)
        # downsample to 4x5
        model = Conv2D(256, (3,3), strides=(4,5), padding='same')(model)
        model = LeakyReLU(alpha=0.2)(model)
        # flatten feature maps
        model = Flatten()(model)
        # dropout
        model = Dropout(0.4)(model)
        # output
        out_layer = Dense(1, activation='sigmoid')(model)
        # define model
        model = Model([in_image, in_label], out_layer)
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model
        
# define the standalone generator model
def define_cmodel_5_generator(latent_dim, n_classes=2, strategy = None):
    if strategy != None:
        with strategy.scope():
            # label input
            in_label = Input(shape=(1,))
            # embedding for categorical input
            lab_input = Embedding(n_classes, 75)(in_label)
            # linear multiplication
            n_nodes = 8 * 8 * 2
            lab_input = Dense(n_nodes)(lab_input)
            # reshape to additional channel
            lab_input = Reshape((8, 8, 2))(lab_input)
            # image generator input
            in_lat = Input(shape=(latent_dim,))
            # foundation for 8x8 image
            n_nodes = 256 * 8 * 8
            gen = Dense(n_nodes)(in_lat)
            gen = LeakyReLU(alpha=0.2)(gen)
            gen = Reshape((8, 8, 256))(gen)
            # merge image gen and label input
            merge = Concatenate()([gen, lab_input])
            # upsample to 16x16
            gen = Conv2DTranspose(256, (3,3), strides=(2,2), padding='same')(merge)
            gen = LeakyReLU(alpha=0.2)(gen)
            # upsample to 32x32
            gen = Conv2DTranspose(256, (3,3), strides=(2,2), padding='same')(gen)
            gen = LeakyReLU(alpha=0.2)(gen)    
            # upsample to 64x64
            gen = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same')(gen)
            gen = LeakyReLU(alpha=0.2)(gen)
            # upsample to 128x128
            gen = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same')(gen)
            gen = LeakyReLU(alpha=0.2)(gen)        
            # upsample to 256x256
            gen = Conv2DTranspose(75, (3,3), strides=(2,2), padding='same')(gen)
            gen = LeakyReLU(alpha=0.2)(gen)       
            # reshape to 128x300 with 128 channels
            gen = Reshape((128, 300, 128))(gen)
            # do another 2D convolution
            gen = Conv2D(64,(3,3), padding = 'same')(gen)
            gen = LeakyReLU(alpha=0.2)(gen)
            # output
            out_layer = Conv2D(1, (3,3), activation='tanh', padding='same')(gen)
            # define model
            model = Model([in_lat, in_label], out_layer)
            return model
    else:
         # label input
        in_label = Input(shape=(1,))
        # embedding for categorical input
        lab_input = Embedding(n_classes, 75)(in_label)
        # linear multiplication
        n_nodes = 8 * 8 * 2
        lab_input = Dense(n_nodes)(lab_input)
        # reshape to additional channel
        lab_input = Reshape((8, 8, 2))(lab_input)
        # image generator input
        in_lat = Input(shape=(latent_dim,))
        # foundation for 8x8 image
        n_nodes = 254 * 8 * 8
        gen = Dense(n_nodes)(in_lat)
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = Reshape((8, 8, 254))(gen)
        # merge image gen and label input
        merge = Concatenate()([gen, lab_input])
        # upsample to 16x16
        gen = Conv2DTranspose(256, (3,3), strides=(2,2), padding='same')(merge)
        gen = LeakyReLU(alpha=0.2)(gen)
        # upsample to 32x32
        gen = Conv2DTranspose(256, (3,3), strides=(2,2), padding='same')(gen)
        gen = LeakyReLU(alpha=0.2)(gen)    
        # upsample to 64x64
        gen = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same')(gen)
        gen = LeakyReLU(alpha=0.2)(gen)
        # upsample to 128x128
        gen = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same')(gen)
        gen = LeakyReLU(alpha=0.2)(gen)        
        # upsample to 256x256
        gen = Conv2DTranspose(75, (3,3), strides=(2,2), padding='same')(gen)
        gen = LeakyReLU(alpha=0.2)(gen)       
        # reshape to 128x300 with 128 channels
        gen = Reshape((128, 300, 128))(gen)
        # do another 2D convolution
        gen = Conv2D(64,(3,3), padding = 'same')(gen)
        gen = LeakyReLU(alpha=0.2)(gen)
        # output
        out_layer = Conv2D(1, (3,3), activation='tanh', padding='same')(gen)
        # define model
        model = Model([in_lat, in_label], out_layer)
        return model

# define the combined generator and discriminator model, for updating the generator
def define_cmodel_5_gan(g_model, d_model, optim = None, strategy = None):
    if strategy != None:
        with strategy.scope():
            # make weights in the discriminator not trainable
            d_model.trainable = False
            # get noise and label inputs from generator model
            gen_noise, gen_label = g_model.input
            # get image output from the generator model
            gen_output = g_model.output
            # connect image output and label input from generator as inputs to discriminator
            gan_output = d_model([gen_output, gen_label])
            # define gan model as taking noise and label and outputting a classification
            model = Model([gen_noise, gen_label], gan_output)
            # compile model
            if optim != None:
                opt = optim
            else:
                opt = Adam(lr=0.0002, beta_1=0.5)
            model.compile(loss='binary_crossentropy', optimizer=opt)
            return model
    else:
        # make weights in the discriminator not trainable
        d_model.trainable = False
        # get noise and label inputs from generator model
        gen_noise, gen_label = g_model.input
        # get image output from the generator model
        gen_output = g_model.output
        # connect image output and label input from generator as inputs to discriminator
        gan_output = d_model([gen_output, gen_label])
        # define gan model as taking noise and label and outputting a classification
        model = Model([gen_noise, gen_label], gan_output)
        # compile model
        if optim != None:
            opt = optim
        else:
            opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model

# This is function for training cGAN network. It uses soft labels for generator and discriminator as well as noised labels for discriminator training
# Also, this code makes sure we use all od the data for training
def train_cgan(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=200, save_epochs = 10, n_batch=128, model_name = "model_1", dir_name = "test", strategy = None):
    
    # if n_batch is odd number
    if n_batch % 2 :
        n_batch += 1
    half_batch      = n_batch // 2

    # number of batches per epoch is determined by number of real data we can use!
    batch_per_epo   = dataset[0].shape[0] // half_batch

    # number of epoch with losses close to 0
    FAULT_EPOCHS    = 3
    bad_epochs      = 0
    last_bad_epoch  = 0
    reset_num       = 0

    # manually enumerate epochs
    i = 0
    while i < n_epochs:

        idx         = np.random.randint(0,dataset[0].shape[0],dataset[0].shape[0])
        save_string = []
        fake_loss   = 0
        real_loss   = 0
        gan_loss    = 0
        start_time  = datetime.now() 
        j = 0
        while j <= batch_per_epo:

            # make sure to collect ALL of the real data samples
            X_real      = dataset[0][idx[j*half_batch:(j+1)*half_batch] if j < batch_per_epo else idx[dataset[0].shape[0]-half_batch:dataset[0].shape[0]]]
            labels_real = dataset[1][idx[j*half_batch:(j+1)*half_batch] if j < batch_per_epo else idx[dataset[0].shape[0]-half_batch:dataset[0].shape[0]]]
            # always generate new nosiy real labels
            y_real      = np.random.uniform(low=0.8, high=1, size=(half_batch,1))
            # update discriminator model weights
            d_loss1, _  = d_model.train_on_batch([X_real, labels_real], y_real)
            # generate 'fake' examples with noisy labels
            [X_fake, labels], y_fake = generate_cfake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
            # prepare points in latent space as input for the generator
            [z_input, labels_input] = generate_clatent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples, make them soft labels by adding noise
            probs   = [0.03, 0.97] # this flips a few labels to 0  
            y_gan   = np.random.choice(2, size = n_batch, p = probs) 
            noise   = np.random.uniform(low = -0.2, high = 0.2, size = n_batch)
            y_gan   = y_gan.astype(float)
            y_gan   += noise
            # make sure no labels are outside of [0,1] range
            y_gan   = np.where(y_gan < 0, 0, y_gan)
            y_gan   = np.where(y_gan > 1, 1, y_gan)
            # update the generator via the discriminator's error
            g_loss  = gan_model.train_on_batch([z_input, labels_input], y_gan)
            # add losses
            fake_loss   += d_loss2
            real_loss   += d_loss1
            gan_loss    += g_loss

            # summarize loss on this batch
            new_loss = '>%d, %d/%d, disc_real=%.3f, disc_fake=%.3f gan=%.3f\n' % (i, j+1, batch_per_epo, d_loss1, d_loss2, g_loss)
            print(new_loss)
            save_string.append(new_loss)
            
            # increment counter
            j += 1

        time_elapsed = datetime.now() - start_time 
        save_string.append('Time elapsed (hh:mm:ss.ms) {}\n'.format(time_elapsed))
        
        # get avg losses per epoch
        real_loss /= batch_per_epo
        fake_loss /= batch_per_epo
        gan_loss  /= batch_per_epo
        
        # summarize loss on this batch
        epcoh_loss = 'epoch %d, real_loss=%.3f, fake_loss=%.3f, gan_loss=%.3f\n' % (i,real_loss, fake_loss, gan_loss)
        save_string.append(epcoh_loss)

        # saving loss values for last epoch
        save_to_txt(lines=save_string, model_name=model_name, dir_name=dir_name)

        # increment how many times this happened
        if (real_loss < 0.2 or fake_loss < 0.2) and (gan_loss > 2 or gan_loss < 0.2) and i>2:
            if last_bad_epoch == i - 1:
                bad_epochs += 1
            else:
                bad_epochs = 1
            last_bad_epoch = i

        # Just to print acc on every epoch
        print_gan_performance(i, g_model, d_model, dataset, latent_dim, folder_path = dir_name)
        
        # save model every save_epochs
        if not (i % save_epochs):
            summarize_cperformance(i, g_model, d_model, dataset, latent_dim, model_name = model_name, folder_path = dir_name)
        
        # redone loading part for this, after FAULT_EPOCHS it loads i//save_epochs * save_epoch
        if not bad_epochs < FAULT_EPOCHS:
            bad_text = f"Bad epochs count is {FAULT_EPOCHS}. We will reset now and load epoch number {(i//save_epochs)*save_epochs}"
            print(bad_text)
            save_to_txt(lines=bad_text, model_name=model_name, dir_name=dir_name)
            bad_epochs = 0
            reset_num += 1
            i = (i//save_epochs)*save_epochs
            d_model, g_model, gan_model = reset_cmodels(i, reset_num, dir_name, strategy = strategy)
        # Make sure to exit 
        if d_model == None or reset_num > 3:
            print(f"Zavrsen trening u epohi {i}, greska!")
            return 
        i += 1
    # at the end of training save both descriminator and generator 
    summarize_cperformance(i, g_model, d_model, dataset, latent_dim, model_name = model_name, folder_path = dir_name)

# This continues GAN training from specific epoch loading generator and discriminator with all data in epoch
# It uses soft labels for generator and discriminator as well as noised labels for discriminator training
def continue_cgan(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=200, save_epochs = 10, start_epoch = 100, n_batch=128, model_name = "model_1", dir_name = "test", strategy = None):
    
    # if n_batch is odd number
    if n_batch % 2 :
        n_batch += 1
    half_batch      = n_batch // 2

    # number of batches per epoch is determined by number of real data we can use!
    batch_per_epo   = dataset[0].shape[0] // half_batch

    # number of epoch with losses close to 0
    FAULT_EPOCHS    = 3
    bad_epochs      = 0
    last_bad_epoch  = 0
    reset_num       = 0

    # manually enumerate epochs
    i = start_epoch
    n_epochs += start_epoch
    while i < n_epochs:

        idx         = np.random.randint(0,dataset[0].shape[0],dataset[0].shape[0])
        save_string = []
        fake_loss   = 0
        real_loss   = 0
        gan_loss    = 0
        start_time  = datetime.now() 
        j = 0
        while j <= batch_per_epo:

            # make sure to collect ALL of the real data samples
            X_real      = dataset[0][idx[j*half_batch:(j+1)*half_batch] if j < batch_per_epo else idx[dataset[0].shape[0]-half_batch:dataset[0].shape[0]]]
            labels_real = dataset[1][idx[j*half_batch:(j+1)*half_batch] if j < batch_per_epo else idx[dataset[0].shape[0]-half_batch:dataset[0].shape[0]]]
            # always generate new nosiy real labels
            y_real      = np.random.uniform(low=0.8, high=1, size=(half_batch,1))
            # update discriminator model weights
            d_loss1, _  = d_model.train_on_batch([X_real, labels_real], y_real)
            # generate 'fake' examples with noisy labels
            [X_fake, labels], y_fake = generate_cfake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
            # prepare points in latent space as input for the generator
            [z_input, labels_input] = generate_clatent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples, make them soft labels by adding noise
            probs   = [0.03, 0.97] # this flips a few labels to 0  
            y_gan   = np.random.choice(2, size = n_batch, p = probs) 
            noise   = np.random.uniform(low = -0.2, high = 0.2, size = n_batch)
            y_gan   = y_gan.astype(float)
            y_gan   += noise
            # make sure no labels are outside of [0,1] range
            y_gan   = np.where(y_gan < 0, 0, y_gan)
            y_gan   = np.where(y_gan > 1, 1, y_gan)
            # update the generator via the discriminator's error
            g_loss  = gan_model.train_on_batch([z_input, labels_input], y_gan)
            # add losses
            fake_loss   += d_loss2
            real_loss   += d_loss1
            gan_loss    += g_loss

            # summarize loss on this batch
            new_loss = '>%d, %d/%d, disc_real=%.3f, disc_fake=%.3f gan=%.3f\n' % (i, j+1, batch_per_epo, d_loss1, d_loss2, g_loss)
            print(new_loss)
            save_string.append(new_loss)
            
            # increment counter
            j += 1

        time_elapsed = datetime.now() - start_time 
        save_string.append('Time elapsed (hh:mm:ss.ms) {}\n'.format(time_elapsed))
        
        # get avg losses per epoch
        real_loss /= batch_per_epo
        fake_loss /= batch_per_epo
        gan_loss  /= batch_per_epo
        

        epcoh_loss = 'epoch %d, real_loss=%.3f, fake_loss=%.3f, gan_loss=%.3f\n' % (i,real_loss, fake_loss, gan_loss)
        save_string.append(epcoh_loss)

        # saving loss values for last epoch
        save_to_txt(lines=save_string, model_name=model_name, dir_name=dir_name)

        # increment how many times this happened
        if (real_loss < 0.2 or fake_loss < 0.2) and (gan_loss > 2 or gan_loss < 0.2) and i>2:
            if last_bad_epoch == i - 1:
                bad_epochs += 1
            else:
                bad_epochs = 1
            last_bad_epoch = i

        # Just to print acc on every epoch
        print_gan_performance(i, g_model, d_model, dataset, latent_dim, folder_path = dir_name)
        
        # save model every save_epochs
        if not (i % save_epochs):
            summarize_cperformance(i, g_model, d_model, dataset, latent_dim, model_name = model_name, folder_path = dir_name)
        
        # redone loading part for this, after FAULT_EPOCHS it loads i//save_epochs * save_epoch
        if not bad_epochs < FAULT_EPOCHS:
            bad_text = f"Bad epochs count is {FAULT_EPOCHS}. We will reset now and load epoch number {(i//save_epochs)*save_epochs}"
            print(bad_text)
            save_to_txt(lines=bad_text, model_name=model_name, dir_name=dir_name)
            bad_epochs = 0
            reset_num += 1
            i = (i//save_epochs)*save_epochs
            d_model, g_model, gan_model = reset_cmodels(i, reset_num, dir_name, strategy = strategy)
        # Make sure to exit 
        if d_model == None or reset_num > 3:
            print(f"Zavrsen trening u epohi {i}, greska!")
            return 
        i += 1
    # at the end of training save both descriminator and generator 
    summarize_cperformance(i, g_model, d_model, dataset, latent_dim, model_name = model_name, folder_path = dir_name)

#######################################################
# Training code with part of the data (like tutorial) #
#######################################################

# This is function for training cGAN network. It uses soft labels for generator and discriminator as well as noised labels for discriminator training
def train_cgan_a(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=200, save_epochs = 10, n_batch=128, model_name = "model_1", dir_name = "test", strategy = None):
    
    # if n_batch is odd number
    if n_batch % 2 :
        n_batch += 1
    half_batch      = n_batch // 2

    # np.round will round number to its closest integer value
    batch_per_epo   = int(np.round(dataset[0].shape[0] / n_batch))

    # number of epoch with losses close to 0
    FAULT_EPOCHS    = 3
    bad_epochs      = 0
    last_bad_epoch  = 0
    reset_num       = 0

    # manually enumerate epochs
    i = 0
    while i < n_epochs:
        # enumerate batches over the training set
        save_string = []
        fake_loss = 0
        real_loss = 0
        gan_loss = 0
        start_time = datetime.now() 
        for j in range(batch_per_epo):
            # get randomly selected 'real' samples
            [X_real, labels_real], y_real = generate_cmeasured_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
            # generate 'fake' examples
            [X_fake, labels], y_fake = generate_cfake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
            # prepare points in latent space as input for the generator
            [z_input, labels_input] = generate_clatent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples, make them soft labels by adding noise
            probs   = [0.03, 0.97] # this flips a few labels to 0  
            y_gan   = np.random.choice(2, size = n_batch, p = probs) 
            noise   = np.random.uniform(low = -0.2, high = 0.2, size = n_batch)
            y_gan   = y_gan.astype(float)
            y_gan   += noise
            # make sure no labels are outside of [0,1] range
            y_gan   = np.where(y_gan < 0, 0, y_gan)
            y_gan   = np.where(y_gan > 1, 1, y_gan)
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
            # add losses
            fake_loss += d_loss2
            real_loss += d_loss1
            gan_loss  += g_loss

            # summarize loss on this batch
            new_loss = '>%d, %d/%d, disc_real=%.3f, disc_fake=%.3f gan=%.3f\n' % (i, j+1, batch_per_epo, d_loss1, d_loss2, g_loss)
            print(new_loss)
            save_string.append(new_loss)

        time_elapsed = datetime.now() - start_time 
        save_string.append('Time elapsed (hh:mm:ss.ms) {}\n'.format(time_elapsed))
        
        # get avg losses per epoch
        real_loss /= batch_per_epo
        fake_loss /= batch_per_epo
        gan_loss  /= batch_per_epo
        
        # summarize loss on this batch
        epcoh_loss = 'epoch %d, real_loss=%.3f, fake_loss=%.3f, gan_loss=%.3f\n' % (i,real_loss, fake_loss, gan_loss)
        save_string.append(epcoh_loss)

        # saving loss values for last epoch
        save_to_txt(lines=save_string, model_name=model_name, dir_name=dir_name)
        
        # increment how many times this happened
        if (real_loss < 0.2 or fake_loss < 0.2) and (gan_loss > 2 or gan_loss < 0.2) and i>2:
            if last_bad_epoch == i - 1:
                bad_epochs += 1
            else:
                bad_epochs = 1
            last_bad_epoch = i

        # Just to print acc on every epoch
        print_gan_performance(i, g_model, d_model, dataset, latent_dim, folder_path = dir_name)

        
        # save model every save_epochs
        if not (i % save_epochs):
            summarize_cperformance(i, g_model, d_model, dataset, latent_dim, model_name = model_name, folder_path = dir_name)
        
        # redone loading part for this, after FAULT_EPOCHS it loads i//save_epochs * save_epoch
        if not bad_epochs < FAULT_EPOCHS:
            bad_text = f"Bad epochs count is {FAULT_EPOCHS}. We will reset now and load epoch number {(i//save_epochs)*save_epochs}"
            print(bad_text)
            save_to_txt(lines=bad_text, model_name=model_name, dir_name=dir_name)
            bad_epochs = 0
            reset_num += 1
            i = (i//save_epochs)*save_epochs
            d_model, g_model, gan_model = reset_cmodels(i, reset_num, dir_name, strategy = strategy)
        # Make sure to exit 
        if d_model == None or reset_num > 3:
            print(f"Zavrsen trening u epohi {i}, greska!")
            return 
        i += 1
    # at the end of training save both descriminator and generator 
    summarize_cperformance(i, g_model, d_model, dataset, latent_dim, model_name = model_name, folder_path = dir_name)

# This continues GAN training from specific epoch loading generator and discriminator.
# It uses soft labels for generator and discriminator as well as noised labels for discriminator training
def continue_cgan_a(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=200, save_epochs = 10, start_epoch = 100, n_batch=128, model_name = "model_1", dir_name = "test", strategy = None):
    
    # if n_batch is odd number
    if n_batch % 2 :
        n_batch += 1
    half_batch      = n_batch // 2

    # np.round will round number to its closest integer value
    batch_per_epo   = int(np.round(dataset[0].shape[0] / n_batch))

    # number of epoch with losses close to 0
    FAULT_EPOCHS    = 3
    bad_epochs      = 0
    last_bad_epoch  = 0
    reset_num       = 0

    # manually enumerate epochs
    i = start_epoch
    n_epochs += start_epoch
    while i < n_epochs:
        # enumerate batches over the training set
        save_string = []
        fake_loss = 0
        real_loss = 0
        gan_loss = 0
        start_time = datetime.now() 
        for j in range(batch_per_epo):
            # get randomly selected 'real' samples
            [X_real, labels_real], y_real = generate_cmeasured_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
            # generate 'fake' examples
            [X_fake, labels], y_fake = generate_cfake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
            # prepare points in latent space as input for the generator
            [z_input, labels_input] = generate_clatent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples, make them soft labels by adding noise
            probs   = [0.03, 0.97] # this flips a few labels to 0  
            y_gan   = np.random.choice(2, size = n_batch, p = probs) 
            noise   = np.random.uniform(low = -0.2, high = 0.2, size = n_batch)
            y_gan   = y_gan.astype(float)
            y_gan   += noise
            # make sure no labels are outside of [0,1] range
            y_gan   = np.where(y_gan < 0, 0, y_gan)
            y_gan   = np.where(y_gan > 1, 1, y_gan)
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
            # add losses
            fake_loss += d_loss2
            real_loss += d_loss1
            gan_loss  += g_loss

            # summarize loss on this batch
            new_loss = '>%d, %d/%d, disc_real=%.3f, disc_fake=%.3f gan=%.3f\n' % (i, j+1, batch_per_epo, d_loss1, d_loss2, g_loss)
            print(new_loss)
            save_string.append(new_loss)

        time_elapsed = datetime.now() - start_time 
        save_string.append('Time elapsed (hh:mm:ss.ms) {}\n'.format(time_elapsed))
        
        # get avg losses per epoch
        real_loss /= batch_per_epo
        fake_loss /= batch_per_epo
        gan_loss  /= batch_per_epo
        
        # summarize loss on this batch
        epcoh_loss = 'epoch %d, real_loss=%.3f, fake_loss=%.3f, gan_loss=%.3f\n' % (i,real_loss, fake_loss, gan_loss)
        save_string.append(epcoh_loss)

        # saving loss values for last epoch
        save_to_txt(lines=save_string, model_name=model_name, dir_name=dir_name)
        
        # increment how many times this happened
        if (real_loss < 0.2 or fake_loss < 0.2) and (gan_loss > 2 or gan_loss < 0.2) and i>2:
            if last_bad_epoch == i - 1:
                bad_epochs += 1
            else:
                bad_epochs = 1
            last_bad_epoch = i

        # Just to print acc on every epoch
        print_gan_performance(i, g_model, d_model, dataset, latent_dim, folder_path = dir_name)

        
        # save model every save_epochs
        if not (i % save_epochs):
            summarize_cperformance(i, g_model, d_model, dataset, latent_dim, model_name = model_name, folder_path = dir_name)
        
        # redone loading part for this, after FAULT_EPOCHS it loads i//save_epochs * save_epoch
        if not bad_epochs < FAULT_EPOCHS:
            bad_text = f"Bad epochs count is {FAULT_EPOCHS}. We will reset now and load epoch number {(i//save_epochs)*save_epochs}"
            print(bad_text)
            save_to_txt(lines=bad_text, model_name=model_name, dir_name=dir_name)
            bad_epochs = 0
            reset_num += 1
            i = (i//save_epochs)*save_epochs
            d_model, g_model, gan_model = reset_cmodels(i, reset_num, dir_name, strategy = strategy)
        # Make sure to exit 
        if d_model == None or reset_num > 3:
            print(f"Zavrsen trening u epohi {i}, greska!")
            return 
        i += 1
    # at the end of training save both descriminator and generator 
    summarize_cperformance(i, g_model, d_model, dataset, latent_dim, model_name = model_name, folder_path = dir_name)


model_name  = "representing_model"
input_shape = (128,300,1)
latent_dim  = 100
EPOCH_NUM   = 200
SAVE_EPOCH  = 2
BATCH_SIZE  = 128
data_path   = DATA_PATH
DIR_PATH    = os.path.join(ROOT_PATH,model_name)

# load data
dataset     = load_training_data(path = data_path)
# plot some real data for comparison
ix = np.random.randint(0, len(dataset[0]), 49)
plot_creal_data([dataset[0][ix],dataset[1][ix]], DIR_PATH)

# To run it on two cards:
strategy = tf.distribute.MirroredStrategy()

# # create descriminator 
d_cmodel_all    = define_cmodel_5_discriminator(input_shape,strategy=strategy)
print_model_summary(d_cmodel_all, "discriminator_" + model_name, dir_name = DIR_PATH)
# # create generator
g_cmodel_all     = define_cmodel_5_generator(latent_dim,strategy=strategy)
print_model_summary(g_cmodel_all, "generator_" + model_name, dir_name = DIR_PATH)
# # create gan model asd adad
gan_cmodel_all   = define_cmodel_5_gan(g_cmodel_all,d_cmodel_all,strategy=strategy)
print_model_summary(gan_cmodel_all, "gan_" + model_name, dir_name = DIR_PATH)
# # train 
train_cgan(g_cmodel_all, d_cmodel_all, gan_cmodel_all, dataset, latent_dim, n_epochs=EPOCH_NUM, save_epochs=SAVE_EPOCH, n_batch=BATCH_SIZE, model_name=model_name, dir_name=DIR_PATH, strategy = strategy)
# Template for training continuation
# START_EPOCH = 200
# d_cmodel_all, g_cmodel_all, gan_cmodel_all = reset_cmodels(START_EPOCH, 0, DIR_PATH, strategy=strategy)
# continue_cgan(g_cmodel_all, d_cmodel_all, gan_cmodel_all, dataset, latent_dim, n_epochs=EPOCH_NUM, save_epochs = SAVE_EPOCH, start_epoch = START_EPOCH, n_batch=BATCH_SIZE, model_name = model_name, dir_name = DIR_PATH, strategy = strategy)