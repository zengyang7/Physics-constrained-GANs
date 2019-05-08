#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 09:54:40 2018

@author: zengyang

Tensorflow, Numpy and Scipy are needed
A mathematic case for constraints build-in cGANs

"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection

from scipy.optimize import curve_fit

# batch size
mb_size = 64

# latent variables
Noise_dim = 30 
num_point = 60

# dimension of data
targ_dim = num_point*2 

# Number of neurals
h_dim1 = 128

# number of auxiliary variable
con_dim = 1 

# dimension of circle
n = 2

# Number of epoch
epoch = 200

# constraint epsilon = cons_value^2
cons_value = 0.2

# weight of penalty term 
lam_constraint = 2


# the function for fit
def fun_fit(t, r, w):
    '''
    The function is to fit trigonometirx function
    t: the parameteric
    r: the amplification coefficient
    w: the initial angel
    '''
    return r*np.cos(t+w)

def next_batch(num, helix, label):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(label))
    np.random.shuffle(idx)
    idx = idx[:num]
    
    helix_shuffle = [helix[i] for i in idx]
    label_shuffle = [label[i] for i in idx]

    return np.asarray(helix_shuffle), np.asarray(label_shuffle)

def plot_(data, n, num_point):
    '''
    Plot the n*(n-1) projection of n-dimensional data
    4*4 plots are presented
    '''
    for i in range(n-1):
        for j in range(i+1, n):
            print('figure x',i+1,'and x',j+1)
            gs = gridspec.GridSpec(4, 4)
            gs.update(wspace=0.05, hspace=0.05)
            for k, sample in enumerate(data):
                ax = plt.subplot(gs[k])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                a = np.reshape(sample, (n, num_point))
                ax.plot(a[i], a[j], marker="o", markersize=1)
                ax.set_xlim(-1.05, 1.05)
                ax.set_ylim(-1.05, 1.05)
            plt.show()
    
    
def plot_cGANs(data, n, real, num_point=80, name=None):
    '''
    Plot the results of cGAN
    '''
    for i in range(n-1):
        for j in range(i+1, n):
            print('figure x',i+1,'and x',j+1)
            gs = gridspec.GridSpec(2, 2)
            gs.update(wspace=0.07, hspace=0.1)
            for k, sample in enumerate(data):
                if k > 3:
                    break
                ax = plt.subplot(gs[k])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                a = np.reshape(sample, (n, num_point))
                b = np.reshape(real[k], (n, num_point))
                ax.plot(a[i], a[j], "o", markersize=2)
                ax.plot(b[i], b[j], "o", markersize=2)
                ax.set_xlim(-1.05, 1.05)
                ax.set_ylim(-1.05, 1.05)
            if name != None:
                namesave = name+'.pdf'
                plt.savefig(namesave)
            plt.show()
            
def plot_cGANs_ring(data, n, real_data, r, name=None):
    '''
    Plot the results of cGAN
    '''
    for i in range(n-1):
        for j in range(i+1, n):
            print('figure x',i+1,'and x',j+1)
            gs = gridspec.GridSpec(2, 2)
            gs.update(wspace=0.03, hspace=0.1)
            for k, sample in enumerate(data):
                if k > 3:
                    break
                ax = plt.subplot(gs[k])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                a = np.reshape(sample, (n, num_point))
                b = np.reshape(real_data[k], (n, num_point))
                ax.plot(a[i], a[j], "o", markersize=2)
                ax.plot(b[i], b[j], "o", markersize=2)
                patches = [Wedge((.0, .0), r[k]+cons_value, 0, 360, width=2*cons_value)]
                p = PatchCollection(patches, alpha=0.3, color='g')
                ax.add_collection(p)
                ax.set_xlim(-1.05, 1.05)
                ax.set_ylim(-1.05, 1.05)
            if name != None:
                namesave = name+'.pdf'
                plt.savefig(namesave)
            plt.show()

def generate_samples(num_samp, num_point):
    '''
    This function is used to generate circles with different radius
    num_samp: the number of samples 
    num_point: the dimenson of hyperparameter
    '''
    
    helix = []
    
    # interval of t
    t_low = 0
    t_up = 2*np.pi
    # generate num_point t in the interval
    t = np.linspace(t_low, t_up, num_point)
    
    # generate num_samp labels (radius of the circle) in [0.5, 0.99]
    label = np.random.uniform(low=0.4, high=0.8, size=(num_samp, 1))
    for i, a in enumerate(label):
        x = a*np.cos(t)
        y = a*np.sin(t)
        data = [x, y]
        data = np.reshape(data, num_point*2)
        helix.append(data)
    return helix, label

# plot samples from 
helix, label = generate_samples(16, num_point)
plot_(helix, 2, num_point)

#-----------------------------------------------------------------------------#
# set GANs
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

# placeholders of discriminator
target = tf.placeholder(tf.float32, shape=[None, targ_dim])
con_v = tf.placeholder(tf.float32, shape=[None, con_dim])

# variables of discriminator
D_W1 = tf.Variable(xavier_init([targ_dim + con_dim, h_dim1]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim1]))
D_W2 = tf.Variable(xavier_init([h_dim1, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))
theta_D = [D_W1, D_W2, D_b1, D_b2]

def log(x):
    return tf.log(x + 1e-8)

def lrelu(x):
    return tf.maximum(x, tf.multiply(x, 0.2))

def discriminator(x, y):
    inputs = tf.concat(axis=1, values=[x, y])
    D_h1 = tf.layers.batch_normalization((tf.matmul(inputs, D_W1) + D_b1), True)
    D_h1 = lrelu(D_h1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob, D_logit

# placeholder of generator
Noise = tf.placeholder(tf.float32, shape=[None, Noise_dim])

# variables of generator
G_W1 = tf.Variable(xavier_init([Noise_dim + con_dim, h_dim1]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim1]))
G_W2 = tf.Variable(xavier_init([h_dim1, targ_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[targ_dim]))

theta_G = [G_W1, G_W2, G_b1, G_b2]

# generator
def generator(Noise, y):
    inputs = tf.concat(axis=1, values=[Noise, y])
    G_h1 = tf.matmul(inputs, G_W1) + G_b1
    G_h1= lrelu(tf.layers.batch_normalization(G_h1, True))
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.tanh(G_log_prob)
    return G_prob

def constraints(x, y):
    # x: mb_size*(2*num_point)
    
    x = tf.reshape(x, [mb_size, 2, num_point])
    x = tf.reduce_sum(tf.square(x), 1) # mb_size*num_point
    
    # change y^2 from mb_size*1 to mb_size*num_point
    y = tf.tile(tf.square(y), [1, num_point])
    
    # (x1^2+x2^2-y^2)^2
    delta_real = tf.square(x - y)
    
    kesi = tf.ones(tf.shape(delta_real))*(cons_value**2)
    delta = delta_real-kesi
    delta = tf.nn.relu(delta)
    delta = tf.reduce_sum(delta, 1)
    delta_real = tf.reduce_sum(delta_real, 1)
    return delta, delta_real
    
def sample_Z(m, n):
    return np.random.randn(m, n)

G_sample = generator(Noise, con_v)
D_real = discriminator(target, con_v)
D_fake = discriminator(G_sample, con_v)

# WGAN-GP
lam = 10
eps = tf.random_uniform([mb_size, 1], minval=0., maxval=1.)
X_inter = eps*target + (1.-eps)*G_sample
grad = tf.gradients(discriminator(X_inter, con_v), [X_inter])[0]
grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
grad_pen = lam * tf.reduce_mean((grad_norm - 1)**2)

delta, delta_real = constraints(G_sample, con_v)
delta_log = tf.log(delta+1)

# WGAN
D_loss = -tf.reduce_mean(D_real) + tf.reduce_mean(D_fake) + grad_pen
G_loss = -tf.reduce_mean(D_fake)+lam_constraint*tf.reduce_mean(delta_log)

D_solver = (tf.train.AdamOptimizer(learning_rate=1e-4).minimize(D_loss, var_list=theta_D))
G_solver = (tf.train.AdamOptimizer(learning_rate=1e-4).minimize(G_loss, var_list=theta_G))

# samples for predict
num_sam = 5000
noise_pred = sample_Z(num_sam, Noise_dim)

helix_test, label_test = generate_samples(num_samp=num_sam, num_point=num_point)
helix_real, con_pred = next_batch(num_sam, helix=helix_test, label=label_test)

helix, label = generate_samples(num_samp=num_sam, num_point=num_point)

record_epoch = []
record_loss_D = []
record_loss_G = []
record_delta = []
record_deviation = []
record_bias = []

t = np.linspace(0, 2*np.pi, num_point)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print('Number of sample: {}'.format(5000))  
    for it in range(epoch+1):
        for iter in range(num_sam//mb_size):
            output_sample = helix[iter*mb_size:(iter+1)*mb_size]
            con_sample = label[iter*mb_size:(iter+1)*mb_size]
            noise_sample = sample_Z(mb_size, Noise_dim)
            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={target: output_sample, Noise: noise_sample, con_v:con_sample})
            _, G_loss_curr, delta_curr = sess.run([G_solver, G_loss, delta_real], feed_dict={target: output_sample, Noise: noise_sample, con_v:con_sample})
                   
        if it % (epoch/20) == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'.format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print('Constraints: {:.4}'.format(np.mean(delta_curr)))
            record_epoch.append(it)
            record_loss_D.append(D_loss_curr)
            record_loss_G.append(G_loss_curr)
            record_delta.append(delta_curr)
            
            # prediction
            target_pred = sess.run(G_sample, feed_dict={Noise: noise_pred, con_v: con_pred})
            
            deviation = 0
            bias = 0
            for i, sample in enumerate(target_pred):
                data_pred = np.reshape(sample, (2, num_point))
                x_pred = data_pred[0]
                y_pred = data_pred[1]
                
                # fit
                popt_x, pcov_x = curve_fit(fun_fit, t, x_pred)
                popt_y, pcov_y = curve_fit(fun_fit, t, y_pred)
                
                # calculate r with fitting
                r = (np.abs(popt_x[0])+np.abs(popt_y[0]))/2
                
                deviation += np.sum((x_pred**2+y_pred**2-r**2)**2)/100
                bias += np.abs(con_pred[i]-r)
            record_bias.append(bias)
            record_deviation.append(deviation)
            # plot the generated data after training 
            name = 'soft_constraint_'+str(it)
            plot_cGANs(target_pred, n, helix_real, num_point=num_point, name='a')
            plot_cGANs_ring(target_pred, n, helix_real, con_pred)
    
            # plot the relationship between epoach and loss_G
#    plt.figure()
#    plt.plot(record_epoch, record_loss_G)
#    plt.show()
#    
#    # plot the relationship between epoach and loss_D
#    plt.figure()
#    plt.plot(record_epoch, record_loss_D)
#    plt.show()
#    
#    # plot the relationship between epoach and deviations
#    plt.figure()
#    plt.plot(record_epoch, record_deviation)
#    
#    # plot the relationship between epoach and biases
#    plt.show()
#    plt.plot(record_epoch, record_bias)
name = 'Cons_Value-GP'+str(cons_value)+'-ep'+str(epoch)+'-lam'+str(lam_constraint)
np.savez_compressed(name, a=record_epoch, b=record_loss_G, 
                    c=record_loss_D, d=record_deviation, e = record_bias, f=target_pred)
