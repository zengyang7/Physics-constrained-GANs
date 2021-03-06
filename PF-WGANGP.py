#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 19:00:16 2018

@author: zengyang
"""
#---------------------------------------------------------
# WGAN-GP for potential flow with constraints

# package
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, time, random
cons_value = 0
tf.reset_default_graph()

# number of mesh
n_mesh = 28

# generate samples
def generate_sample(n, parameter):
    ''' 
    generate samples of potential flow
    two kinds of potential flows are used : Uniform and source
    Uniform: F1(z) = V*exp(-i*alpha)*z
    source:  F2(z) = m/(2*pi)*log(z)
    x: interval of x axis
    y: interval of y axis
    n: number size of mesh
    parameter: V, alpha, m
    output: u, v the velocity of x and y direction
    '''
    # mesh
    # mesh
    x = [-0.5, 0.5]
    y = [-0.5, 0.5]
    x_mesh = np.linspace(x[0], x[1], int(n))
   
    y_mesh = np.linspace(y[0], y[1], int(n))
    
    X, Y = np.meshgrid(x_mesh, y_mesh)  
    U = []
    
    for i, p in enumerate(parameter):
        V = p[0]
        alpha  = p[1]
        m = p[2]
        
        # velocity of uniform
        u1 = np.ones([n, n])*V*np.cos(alpha)
        v1 = np.ones([n, n])*V*np.sin(alpha)
        
        # velocity of source
        # u2 = m/2pi * x/(x^2+y^2)
        # v2 = m/2pi * y/(x^2+y^2)
        # the source is at (0.8,0)
        u2 = m/(2*np.pi)*(X-0.8)/((X-0.8)**2+Y**2)
        v2 = m/(2*np.pi)*Y/((X-0.8)**2+Y**2)
        
        ur = m/(2*np.pi)/np.sqrt(X**2+Y**2)
        
        u = u1+u2
        v = v1+v2
        
        U_data = np.zeros([n, n, 2])
        U_data[:, :, 0] = u
        U_data[:, :, 1] = v
        U.append(U_data)
    return X, Y, np.asarray(U), ur

def plot_samples(X, Y, U, name=None):
    '''
    plot the samples
    '''
    figu, axsu = plt.subplots(4, 4, figsize=(15, 6))
    figv, axsv = plt.subplots(4, 4, figsize=(15, 6))
    figu.subplots_adjust(hspace = .05, wspace=.01)
    figv.subplots_adjust(hspace = .05, wspace=.01)
    axsu = axsu.ravel()
    axsv = axsv.ravel()
    data_num = len(U)
    num_plot = np.min([data_num, 16])
    for i in range(num_plot):
        sample = U[i]
        axsu[i].contourf(X, Y, sample[:,:,0])
        axsu[i].set_xticklabels([])
        axsu[i].set_yticklabels([])
        
        axsv[i].contourf(X, Y, sample[:,:,1])

        axsv[i].set_xticklabels([])
        axsv[i].set_yticklabels([])
    if name == None:
        plt.show()
    else:
        nameu = name+'u.png'
        namev = name+'v.png'
        figu.savefig(nameu)
        figv.savefig(namev)
    plt.close()

# setting for training sample
n_sam = 2000
V_mu, V_sigma = 0.5, 0.1
alpha_mu, alpha_sigma = np.pi/4, np.pi/20
m_mu, m_sigma = 1, 0.2

samples = np.zeros([n_sam, 3])

V_sample = np.random.normal(V_mu, V_sigma, n_sam)
alpha_sample = np.random.normal(alpha_mu, alpha_sigma, n_sam)
m_sample = np.random.normal(m_mu, m_sigma, n_sam)

samples[:,0] = V_sample
samples[:,1] = alpha_sample
samples[:,2] = m_sample

X, Y, U, ur = generate_sample(n=n_mesh, parameter=samples)
plot_samples(X, Y, U)

d_x = X[:,1:]-X[:,0:-1]
d_y = Y[1:,:]-Y[0:-1,:]
#-----------------------------------------------------------------------------#
#GANs
n_label = 3

def next_batch(num, labels, U):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(labels))
    np.random.shuffle(idx)
    idx = idx[:num]
    
    U_shuffle = [U[i] for i in idx]
    label_shuffle = [labels[i] for i in idx]

    return np.asarray(U_shuffle), np.asarray(label_shuffle)

# parameters of GANs
    
# leak_relu
def lrelu(X, leak=0.2):
    f1 = 0.5*(1+leak)
    f2 = 0.5*(1+leak)
    return f1*X+f2*tf.abs(X)

# G(z)
def generator(z, y_label, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        # initializer
        w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        b_init = tf.constant_initializer(0.0)

        # concat layer
        # z_ = np.random.normal(0, 1, (batch_size, 1, 1, 30)),
        # y_label.shape = [batch_size, 1, 1, 3]
        cat1 = tf.concat([z, y_label], 3)

        # 1st hidden layer
        # output.shape = (kernal.shape-1)*stride+input.shape
        # deconv1.shape = [batch_size, output.shape, **, channel]
        deconv1 = tf.layers.conv2d_transpose(cat1, 256, [7, 7], strides=(1, 1), padding='valid', 
                                             kernel_initializer=w_init, bias_initializer=b_init)
        lrelu1 = lrelu(tf.layers.batch_normalization(deconv1, training=isTrain), 0.2)

        # 2nd hidden layer
        deconv2 = tf.layers.conv2d_transpose(lrelu1, 128, [5, 5], strides=(2, 2), padding='same', 
                                             kernel_initializer=w_init, bias_initializer=b_init)
        lrelu2 = lrelu(tf.layers.batch_normalization(deconv2, training=isTrain), 0.2)

        # output layer
        deconv3 = tf.layers.conv2d_transpose(lrelu2, 2, [5, 5], strides=(2, 2), padding='same', 
                                             kernel_initializer=w_init, bias_initializer=b_init)
        o = tf.nn.tanh(deconv3)

        return o

# D(x)
def discriminator(x, y_fill, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # initializer
        w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        b_init = tf.constant_initializer(0.0)

        # concat layer
         
        cat1 = tf.concat([x, y_fill], 3)

        # 1st hidden layer
        conv1 = tf.layers.conv2d(cat1, 128, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        lrelu1 = lrelu(conv1, 0.2)

        # 2nd hidden layer
        conv2 = tf.layers.conv2d(lrelu1, 256, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

        # output layer
        conv3 = tf.layers.conv2d(lrelu2, 1, [7, 7], strides=(1, 1), padding='valid', kernel_initializer=w_init)
        o = tf.nn.sigmoid(conv3)

        return o, conv3

batch_size = 100
   
def constraints(x,dx,dy):
    '''
    This function is the constraints of potentianl flow, 
    L Phi = 0, L is the laplace calculator
    Phi is potential function
    '''
    # x.shape [batch_size, img_size, img_size, 2]
    u = tf.slice(x, [0,0,0,0], [batch_size, n_mesh, n_mesh, 1])
    v = tf.slice(x, [0,0,0,1], [batch_size, n_mesh, n_mesh, 1])
    
    u = tf.reshape(u,[batch_size, n_mesh, n_mesh])
    v = tf.reshape(v,[batch_size, n_mesh, n_mesh])
    
    u_left = tf.slice(u, [0,0,0], [batch_size, n_mesh, n_mesh-1])
    u_right = tf.slice(u, [0,0,1], [batch_size, n_mesh, n_mesh-1])
    d_u = tf.divide(tf.subtract(u_right,u_left), dx)
    
    v_up = tf.slice(v, [0,0,0], [batch_size, n_mesh-1, n_mesh])
    v_down = tf.slice(v, [0,1,0], [batch_size, n_mesh-1, n_mesh])
    d_v = tf.divide(tf.subtract(v_down,v_up), dy)
    
    delta_u = tf.slice(d_u, [0,1,0],[batch_size, n_mesh-1, n_mesh-1])
    delta_v = tf.slice(d_v, [0,0,1],[batch_size, n_mesh-1, n_mesh-1])
    
    delta = tf.square(delta_u+delta_v)
    
    # value average to reduce the influence of mesh size
    delta = tf.reduce_mean(delta, 2)
    delta_real_ = tf.reduce_mean(delta, 1)
    
    kesi = tf.ones(tf.shape(delta_real_))*(cons_value)
    delta_lose_ = delta_real_ - kesi
    delta_lose_ = tf.nn.relu(delta_lose_)
    return delta_lose_, delta_real_

lam_cons = 1

# preprocess
img_size = n_mesh
#onehot = np.eye(3)

# lr = 0.0002
train_epoch = 2

global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(0.0002, global_step, 500, 0.95, staircase=True)

# variables : input
x = tf.placeholder(tf.float32, shape=(None, img_size, img_size, 2))
z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100))
y_label = tf.placeholder(tf.float32, shape=(None, 1, 1, n_label))
y_fill = tf.placeholder(tf.float32, shape=(None, img_size, img_size, n_label))
isTrain = tf.placeholder(dtype=tf.bool)

dx = tf.placeholder(tf.float32, shape=(None, img_size, img_size-1))
dy = tf.placeholder(tf.float32, shape=(None, img_size-1, img_size))

# networks : generator
G_z = generator(z, y_label, isTrain)

# networks : discriminator
D_real, D_real_logits = discriminator(x, y_fill, isTrain)
D_fake, D_fake_logits = discriminator(G_z, y_fill, isTrain, reuse=tf.AUTO_REUSE)
delta_lose, delta_real = constraints(G_z, dx, dy)

# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

lam_GP = 10

# WGAN-GP
eps = tf.random_uniform([batch_size, 1], minval=0., maxval=1.)
eps = tf.reshape(eps,[batch_size, 1, 1, 1])
eps = eps * np.ones([batch_size, img_size, img_size, 2])
X_inter = eps*x + (1.-eps)*G_z
grad = tf.gradients(discriminator(X_inter, y_fill, isTrain, reuse=tf.AUTO_REUSE), [X_inter])[0]
grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
grad_pen = lam_GP * tf.reduce_mean((grad_norm - 1)**2)


# loss for each network
D_loss_real = -tf.reduce_mean(D_real_logits)
D_loss_fake = tf.reduce_mean(D_fake_logits)
D_loss = D_loss_real + D_loss_fake + grad_pen
delta_loss = tf.reduce_mean(delta_lose)
G_loss_only = -tf.reduce_mean(D_fake_logits)
G_loss = G_loss_only + lam_cons*(delta_loss) 
#+ lam*tf.reduce_mean(cons_value)

# record loss function for each network
D_loss_real_record = -tf.reduce_mean(tf.log(D_real))
D_loss_fake_record = tf.reduce_mean(tf.log(1. - D_fake))
D_loss_record = D_loss_real_record + D_loss_fake_record
G_loss_record = -tf.reduce_mean(tf.log(D_fake))

D_loss_real_record = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1, 1, 1])))
D_loss_fake_record = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 1, 1, 1])))
D_loss_record = D_loss_real_record + D_loss_fake_record
G_loss_record = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 1, 1, 1]))) 



# optimizer for each network
    
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    optim = tf.train.AdamOptimizer(lr, beta1=0.5)
    D_optim = optim.minimize(D_loss, global_step=global_step, var_list=D_vars)
    # D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(G_loss, var_list=G_vars)

#with tf.name.scope('Grad'):
#    grads_D = tf.gradients(D_loss, D_vars)
#    grads_G = tf.gradients(G_loss, G_vars)
#    
#for grad_d, var_d in grads_D:
#    tf.summary.histogram(var_d.name + '/gradient', grad_d)
#for grad_g, var_g in grads_G:
#    tf.summary.histogram(var_g.name + '/gradient', grad_g)
#merged_summary_op = tf.summary.merge_all()
## open session and initialize all variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# MNIST resize and normalization
# train_set = tf.image.resize_images(mnist.train.images, [img_size, img_size]).eval()
# train_set = (train_set - 0.5) / 0.5  # normalization; range: -1 ~ 1
train_set = (U-np.max(U)/2)/np.max(U)
train_label = samples

root = './Potentialflow-results/'

if not os.path.isdir(root):
    os.mkdir(root)

# save model and all variables
saver = tf.train.Saver()

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []
train_hist['delta_real'] = []
train_hist['delta_lose'] = []
train_hist['prediction'] = []


# training-loop
np.random.seed(int(time.time()))
print('training start!')
start_time = time.time()

d_x_ = np.tile(d_x, (batch_size, 1)).reshape([batch_size, n_mesh, n_mesh-1])
d_y_ = np.tile(d_y, (batch_size, 1)).reshape([batch_size, n_mesh-1, n_mesh])

for epoch in range(train_epoch+1):
    G_losses = []
    D_losses = []
    delta_real_record = []
    delta_lose_record = []
    epoch_start_time = time.time()
    shuffle_idxs = random.sample(range(0, train_set.shape[0]), train_set.shape[0])
    shuffled_set = train_set[shuffle_idxs]
    shuffled_label = train_label[shuffle_idxs]
    for iter in range(shuffled_set.shape[0] // batch_size):
        # update discriminator
        x_ = shuffled_set[iter*batch_size:(iter+1)*batch_size]
        y_label_ = shuffled_label[iter*batch_size:(iter+1)*batch_size].reshape([batch_size, 1, 1, n_label])
        y_fill_ = y_label_ * np.ones([batch_size, img_size, img_size, n_label])
        z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))
        
        loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, y_fill: y_fill_, y_label: y_label_, isTrain: True})
        
        # update generator
        z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))
        
        #####  test the dimension
        #_, y_ = next_batch(batch_size, samples, U)
        #y_label_ = onehot[y_.astype(np.int32)].reshape([batch_size, 1, 1, 3])
        
        # y_fill_.shape = [batch_size, img_size, img_size, 3]
        #y_fill_ = y_label_ * np.ones([batch_size, img_size, img_size, 3])
        loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, x: x_, y_fill: y_fill_, y_label: y_label_, dx:d_x_, dy:d_y_, isTrain: True})

        errD_fake = D_loss_fake_record.eval({z: z_, y_label: y_label_, y_fill: y_fill_, isTrain: False})
        errD_real = D_loss_real_record.eval({x: x_, y_label: y_label_, y_fill: y_fill_, isTrain: False})
        errG = G_loss_record.eval({z: z_, y_label: y_label_, y_fill: y_fill_, dx:d_x_, dy:d_y_, isTrain: False})
        errdelta_real = delta_real.eval({z: z_, y_label: y_label_, y_fill: y_fill_, dx:d_x_, dy:d_y_, isTrain: False})
        errdelta_lose = delta_lose.eval({z: z_, y_label: y_label_, y_fill: y_fill_, dx:d_x_, dy:d_y_, isTrain: False})
        
        D_losses.append(errD_fake + errD_real)
        G_losses.append(errG)
        delta_real_record.append(errdelta_real)
        delta_lose_record.append(errdelta_lose)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f, delta: %.3f' % 
          ((epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses), np.mean(delta_real_record)))
    train_hist['D_losses'].append(np.mean(D_losses))
    train_hist['G_losses'].append(np.mean(G_losses))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
    train_hist['delta_real'].append(np.mean(delta_real_record))
    train_hist['delta_lose'].append(np.mean(delta_lose_record))
    ### need change every time, PF: potential flow, 
    name = root + 'PF-WGANGP-cons0-lam1-va'+str(epoch)
    
    z_pred = np.random.normal(0, 1, (16, 1, 1, 100))
    y_label_pred = shuffled_label[0:16].reshape([16, 1, 1, n_label])
    #summary = sess.run([merged_summary_op],
                       #feed_dict={x: x_, z: z_, y_fill: y_fill_, y_label: y_label_, dx:d_x_, dy:d_y_, isTrain: False})
    prediction = G_z.eval({z:z_pred, y_label:y_label_pred, isTrain: False})
    prediction = prediction*np.max(U)+np.max(U)/2
    train_hist['prediction'].append(prediction)
    
    plot_samples(X, Y, prediction)
    
end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

name_data = root+'PF-WGANGP-cons0-lam1-va'
np.savez_compressed(name_data, a=train_hist, b=per_epoch_ptime)
save_model = name_data+'.ckpt'
save_path = saver.save(sess, save_model)
