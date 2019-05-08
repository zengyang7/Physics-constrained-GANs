#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 14:39:00 2018

@author: zengyang
"""

# GANs for physical constraints -- high dimensional helix function

###----------------------------------------------------###
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D 
#import matplotlib.gridspec as gridspec
import time

# the rotation matrix for rotation
import rotation_matrix as rm

###-----------------------------------------------------###
# parameter for training

n = 4
num_point = 150
mb_size = 64
Noise_dim = 10*n
targ_dim = num_point*n
h_dim1 = 128
h_dim2 = 64

# the original plane for rotation
V = np.array([[0, 0, 0, 0],[0, 1, 1, 1], [0, 1, 2, 3]])
# the angel for rotation
theta = np.pi/4

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

#def plot_(data, n, num_point=150):
#    for i in range(n-1):
#        for j in range(i+1, n):
#            print('figure x',i+1,'and x',j+1)
#            #fig = plt.figure(figsize=(4, 4))
#            gs = gridspec.GridSpec(4, 4)
#            gs.update(wspace=0.05, hspace=0.05)
#            for k, sample in enumerate(data):
#                ax = plt.subplot(gs[k])
#                plt.axis('off')
#                ax.set_xticklabels([])
#                ax.set_yticklabels([])
#                ax.set_aspect('equal')
#                a = np.reshape(sample, (n, num_point))
#                ax.plot(a[i], a[j], 'o', markersize=1)
#                #ax.plot(a[i], a[j],'o')
#                #ax.set_xlim(-1.05, 1.05)
#                #ax.set_ylim(-1.05, 1.05)
#            plt.show()
#            
def generate_samples_3D(num_samp, num_point=150):
    '''
    This function is used to generate high dimensional ellipse samples from 3D
    
    num_samp : the number of samples 
    n        : the dimension of ellipse
    num_point: the dimenson of hyperparameter
    '''
    
    helix = []
    
    n_helix = num_point
    t_low = 0
    t_up = 4*np.pi
    t = np.linspace(t_low, t_up, n_helix)
    
    # Generate num_samp groups of parameters, the interval of parameters are in
    # [1, 2]
    label = np.random.uniform(low=1.0, high=2.0, size=(num_samp, 3))
    
    for i, a in enumerate(label):
        data = []
        x = a[0]*np.cos(t)
        y = a[1]*np.sin(t)
        z = a[2]*t
        data = [x, y, z]
        helix.append(data)
    return np.asarray(helix), np.asarray(label)

def transfer3D_to_nD(helix, n, V, theta):
    ''' 
    This function transfer the 3D helix to nD dimension
    helix: the data generated from generate_samples_3D
    helix: num_samp * 3 * num_point array
    n    : the dimension of the space
    V    : the origin of rotation
    theta: the angel of rotation
    '''
    # calculate rotation matrix based on origin and angel
    M = rm.rotation_matrix(V, theta)
    
    # add zero vector when n is more than 3
    zero = np.zeros([helix.shape[0], n-helix.shape[1], helix.shape[2]])
    helix = np.concatenate((helix, zero), axis=1)
    
    # add one vector due to rotation matrix is 1 dimensional more than n
    one = np.ones([helix.shape[0], 1, helix.shape[2]])
    helix = np.concatenate((helix, one), axis=1)
    
    helix_s = []
    for i, a in enumerate(helix):
        # rotation
        b = np.dot(np.transpose(a),M)
        
        # The list column is all one for rotation, discard it
        b = b[...,:-1]
        
        b = np.reshape(np.transpose(b), helix.shape[2]*n)
        helix_s.append(b)
    return np.asarray(helix_s), M


def transfernD_to_3D(helix_s, n, num_point, M):
    '''
    This function transfer the nD helix to 3D dimension
    '''
    helix = []
    invM = np.linalg.inv(M)
    for i, a in enumerate(helix_s):
        a = np.reshape(a, (n, num_point))
        one = np.ones([num_point, 1])
        a = np.concatenate((np.transpose(a), one), axis=1)
        b = np.dot(a, invM)
        b = np.transpose(b)
        b = np.reshape(b[:-2,...], 3*num_point)
        helix.append(b)
    return np.asarray(helix)


helix, label = generate_samples_3D(16, num_point=num_point)
#plot_(helix, 3)
# dimension of high space

helix_s, M = transfer3D_to_nD(helix, n, V, theta)
plot_(helix_s, 4)

#----------------------------------------------------------------#
# Define GANs
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def log(x):
    return tf.log(x + 1e-8)

def lrelu(x):
    return tf.maximum(x, tf.multiply(x, 0.2))

# placeholders of discriminator
x = tf.placeholder(tf.float32, shape=[None, targ_dim])
    
# variables of discriminator
D_W1 = tf.Variable(xavier_init([targ_dim , h_dim1]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim1]))
D_W2 = tf.Variable(xavier_init([h_dim1, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))
theta_D = [D_W1, D_W2, D_b1, D_b2]

def discriminator(x):
    D_h1 = lrelu(tf.matmul(x, D_W1) + D_b1)
    D_prob = (tf.matmul(D_h1, D_W2) + D_b2)
    return D_prob

# placeholder of generator
Noise = tf.placeholder(tf.float32, shape=[None, Noise_dim])

# variables of generator
G_W1 = tf.Variable(xavier_init([Noise_dim, h_dim1]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim1]))
G_W2 = tf.Variable(xavier_init([h_dim1, targ_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[targ_dim]))

theta_G = [G_W1, G_W2, G_b1, G_b2]

def generator(Noise):
    inputs = Noise
    G_h1 = lrelu(tf.matmul(inputs, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.tanh(G_log_prob)
    return G_prob


def sample_Z(m, n):
    return np.random.randn(m, n)#np.random.uniform(-1., 1., size=[m, n])

G_sample = generator(Noise)
D_real = discriminator(x)
D_fake = discriminator(G_sample)

# WGAN-GP
lam = 10
eps = tf.random_uniform([mb_size, 1], minval=0., maxval=1.)
X_inter = eps*x + (1.-eps)*G_sample
grad = tf.gradients(discriminator(X_inter), [X_inter])[0]
grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
grad_pen = lam * tf.reduce_mean((grad_norm - 1)**2)

D_loss = -tf.reduce_mean(D_real) + tf.reduce_mean(D_fake)+grad_pen
G_loss = -tf.reduce_mean(D_fake)

D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(D_loss, var_list=theta_D))
G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(G_loss, var_list=theta_G))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epoch = 200
num_sam = 50000
np.random.seed(1)
helix, label = generate_samples_3D(num_sam)
helix_s, M = transfer3D_to_nD(helix, n, V, theta)
print(len(helix_s))

# normalize
h_low = np.min(helix_s)
h_up = np.max(helix_s)
helix_s = 2*(helix_s-h_low)/(h_up-h_low)-1.0
Generated_samples = []
D_loss_record = []
G_loss_record = []

start_time = time.time()

for it in range(epoch+1):
    for iter in range(num_sam//mb_size):
        output_sample = helix_s[iter*mb_size:(iter+1)*mb_size]
        con_sample = label[iter*mb_size:(iter+1)*mb_size]
        
        np.random.seed(int(time.time()))
        noise_sample = sample_Z(mb_size, Noise_dim)
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={x: output_sample, Noise: noise_sample})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={x: output_sample, Noise: noise_sample})

    if it % (epoch/10) == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'.format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()
        np.random.seed(1)
        z = sample_Z(200, Noise_dim)
        Generated_sample = sess.run(G_sample, feed_dict={Noise: z})
        Generated_sample = (Generated_sample+1.0)*(h_up-h_low)/2+h_low
        Generated_samples.append(Generated_sample)
        D_loss_record.append(D_loss_curr)
        G_loss_record.append(G_loss_curr)

name_data =  'HighdimensionalHelix'+'-ep'+str(epoch)
np.savez_compressed(name_data, a=Generated_samples, b=D_loss_record, c=G_loss_record)
#n_pred = 16
#helix_real,_= next_batch(n_pred, helix=helix_s, label=label)
#noise_pred = sample_Z(n_pred, Noise_dim)
#target_pred = sess.run(G_sample, feed_dict={Noise: noise_pred})
#target_pred = (target_pred+1.0)*(h_up-h_low)/2+h_low
#plot_(target_pred, n)
