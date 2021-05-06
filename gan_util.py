# -*- coding: utf-8 -*-
"""
Created on Thu Sat Apr  24 13:10:49 2021

Util functions for cGAN training code

@author: milan.medic
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

##############################################################################
#                     COMMON UTILITY FUNCTIONS CODE                          #
##############################################################################

# Update loss value to txt document
def save_to_txt(lines, model_name = "model_1", dir_name = "loss_folder"):
    
    # Make sure to save file into correct folder
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        
    filepath = os.path.join(dir_name,model_name+'_loss.txt')
    with open(filepath, 'a') as f:
        f.writelines(lines)    

# This method print summary to console, and saves image of model arhitecture
def print_model_summary(model, model_name, dir_name):
    
    # summarize the model
    model.summary()
    #check if directory exists, if not create it
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    # plot the model
    plot_model(model, to_file = os.path.join(dir_name,model_name +'.png'), show_shapes = True, show_layer_names = True)

##############################################################################
#                              CGAN FUNCTIONS CODE                           #
##############################################################################

# This methos saves png of generated images/data with conditional GAN
def save_cplot(examples, epoch, dir_name ,title = "epoch", model_name = None ,n=7):
    
    # Make sure to save file into correct folder
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    # separate data
    images, labels = examples
    lbls = []
    for label in labels:
        if label == 0:
            lbls.append("Starting") 
        else:
            lbls.append("Walking") 
    # plot images
    plt.figure(figsize=(10,7))
    plt.subplots_adjust( wspace=0.2, hspace=0.4)
    plt.suptitle(title+ f"{epoch}", fontsize = 22)
    for i in range(n*n):
        plt.subplot(7,7,i+1)
        plt.imshow(images[i], interpolation='nearest', aspect='auto')
        plt.axis("off")
        plt.title(lbls[i], color = 'green')
    
    # save plot to file
    if model_name != None:
        filename = f"{model_name} generated_plot_e{epoch}.png"
    else:
        filename = 'generated_plot_e%03d.png' % (epoch)
        
    filename = os.path.join(dir_name,filename)
    plt.savefig(filename)
    plt.close()

def print_gan_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=150,folder_path = "folder"):
    # prepare real samples
    X_real, y_real = generate_cmeasured_samples_print(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_cfake_samples_print(g_model, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    
    # summarize discriminator performance
    save_string = '%d%%>Accuracy real: %.0f%%, fake: %.0f%%\n' % (epoch,acc_real*100, acc_fake*100)
    print(save_string)

    # save plot
    save_cplot(x_fake, epoch,folder_path)

    return save_string

    
# evaluate the discriminator, plot generated images, save generator model
# This method is the same for both!
def summarize_cperformance(epoch, g_model, d_model, dataset, latent_dim, n_samples=150, model_name = 'model_1', folder_path = "folder"):
    
    # prepare real samples
    X_real, y_real = generate_cmeasured_samples_print(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_cfake_samples_print(g_model, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    
    # summarize discriminator performance
    save_string = '>Accuracy real: %.0f%%, fake: %.0f%%\n' % (acc_real*100, acc_fake*100)
    print(save_string)
    
    # Make sure to save file into correct folder
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    # save the generator model tile file
    generator_folder = os.path.join(folder_path,"generators")
    
    if not os.path.exists(generator_folder):
        os.mkdir(generator_folder)
        
    # save to txt doc
    save_to_txt(save_string,model_name,folder_path)
    
    # save plot
    save_cplot(x_fake, epoch,folder_path)
    
    filename = os.path.join(generator_folder,f'generator_model_{epoch}.h5')
    g_model.save(filename)
    
    d_model.trainable = True
    filename = os.path.join(generator_folder,f'discriminator_model_{epoch}.h5')
    d_model.save(filename)
    d_model.trainable = False

# Method for plotting data and saving it into directory: dir_name
def plot_creal_data(data, dir_name, n = 7):
    
    # Make sure to save file into correct folder
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    # separate data
    images, labels = data
    lbls = []
    for label in labels:
        if label == 0:
            lbls.append("Starting")
        else:
            lbls.append("Walking")
    # plot images
    plt.figure(figsize=(10,7))
    plt.subplots_adjust( wspace=0.2, hspace=0.4)
    plt.suptitle("Measured data", fontsize = 22)
    for i in range(n*n):
        plt.subplot(7,7,i+1)
        plt.imshow(images[i], interpolation='nearest', aspect='auto')
        plt.axis("off")
        plt.title(lbls[i], color = 'green')
    filename = "generated_measured_data"
    filename = os.path.join(dir_name,filename)
    plt.savefig(filename)
    plt.close()

#####################
# DATA LOADING CODE #
#####################


def load_training_data(path, splitting = False, splitting_coef = 0.9):
    
    with open(path, "rb" ) as f:
        data_dict = pickle.load(f)
        
    data    = data_dict['data']
    labels  = data_dict['labels']
    
    data    = np.expand_dims(np.array(data),-1)
    labels  = np.array(labels)

    print(f'Size of dataset: {len(data)}')
    return [data, labels]

def generate_clatent_points(latent_dim, n_samples, n_classes = 2):
    # generate points in the latent space
    x_input = np.random.normal(size = latent_dim * n_samples) # Use Gaussian distribution
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    # probabilities are calculated and stored in "probabilities.mat"
    probs   = [0.3133532548192418, 0.6866467451807582]
    labels  = np.random.choice(n_classes, size = n_samples, p = probs)
    return [z_input, labels]

# This function generates measured samples (from dataset) with hard labels [0,1]
# This us used for ploting and printing accuracy of discriminator
def generate_cmeasured_samples_print(dataset, n_samples):
    # split into images and labels
    images, labels = dataset
    # choose random instances
    ix = np.random.randint(0, images.shape[0], n_samples)
    # select images and labels
    X, labels = images[ix], labels[ix]
    # generate class labels
    y = np.ones((n_samples, 1))
    return [X, labels], y

# This function generates measured samples with soft labels used for GAN training!
def generate_cmeasured_samples(dataset, n_samples):
    # split into images and labels
    images, labels = dataset
    # choose random instances
    ix = np.random.randint(0, images.shape[0], n_samples)
    # select images and labels
    X, labels = images[ix], labels[ix]
    # generate class labels
    y = np.random.uniform(low=0.8, high=1, size=(n_samples,1))
    return [X, labels], y


# use generator to generate n fake examples for printing, with hard labels
def generate_cfake_samples_print(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input, labels_input = generate_clatent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict([z_input, labels_input])
    # create class labels
    y = np.zeros((n_samples, 1))
    return [images, labels_input], y

# use the generator to generate n fake examples, with class labels
def generate_cfake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input, labels_input = generate_clatent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict([z_input, labels_input])
    # create class labels
    y = np.random.uniform(low=0, high=0.2, size=(n_samples,1))
    return [images, labels_input], y