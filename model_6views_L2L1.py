# This file is specific for multi-view data with 6 views!

import tensorflow as tf
import numpy as np
from utils.Net_UFRL import Net_UFRL
from utils.VFRL_L2L1 import VFRL
import math
from sklearn.utils import shuffle


def model(X1, X2, X3, X4, X5, X6, gt, para_beta, dims, act, lr, epochs, batch_size, k_nn_vfrl):

    net_ufrl1 = Net_UFRL(1, dims[0], act[0])
    net_ufrl2 = Net_UFRL(2, dims[1], act[1])
    net_ufrl3 = Net_UFRL(3, dims[2], act[2])
    net_ufrl4 = Net_UFRL(4, dims[3], act[3])
    net_ufrl5 = Net_UFRL(5, dims[4], act[4])
    net_ufrl6 = Net_UFRL(6, dims[5], act[5])

    H = np.zeros([X1.shape[0], dims[0][0]])
    G1 = np.zeros(X1.shape)
    G2 = np.zeros(X2.shape)
    G3 = np.zeros(X3.shape)
    G4 = np.zeros(X4.shape)
    G5 = np.zeros(X5.shape)
    G6 = np.zeros(X6.shape)

    with tf.variable_scope("H"):
        h_input = tf.Variable(xavier_init(batch_size, dims[0][0]), name='LatentSpaceData')
        h_list = tf.trainable_variables()

    fea1_latent = tf.placeholder(np.float32, [None, X1.shape[1]])
    fea2_latent = tf.placeholder(np.float32, [None, X2.shape[1]])
    fea3_latent = tf.placeholder(np.float32, [None, X3.shape[1]])
    fea4_latent = tf.placeholder(np.float32, [None, X4.shape[1]])
    fea5_latent = tf.placeholder(np.float32, [None, X5.shape[1]])
    fea6_latent = tf.placeholder(np.float32, [None, X6.shape[1]])

    loss_ufrl_mse = net_ufrl1.loss_mse_reconstruction(h_input, fea1_latent) + net_ufrl2.loss_mse_reconstruction(h_input, fea2_latent) \
              + net_ufrl3.loss_mse_reconstruction(h_input, fea3_latent) + net_ufrl4.loss_mse_reconstruction(h_input, fea4_latent) \
              + net_ufrl5.loss_mse_reconstruction(h_input, fea5_latent) + net_ufrl6.loss_mse_reconstruction(h_input, fea6_latent)

    update_ufrl = tf.train.AdamOptimizer(lr[0]).minimize(loss_ufrl_mse, var_list=[net_ufrl1.netpara, net_ufrl2.netpara,
                                                                          net_ufrl3.netpara, net_ufrl4.netpara,
                                                                          net_ufrl5.netpara, net_ufrl6.netpara])
    update_h = tf.train.AdamOptimizer(lr[1]).minimize(loss_ufrl_mse, var_list=h_list)

    g1 = net_ufrl1.get_x(h_input)
    g2 = net_ufrl2.get_x(h_input)
    g3 = net_ufrl3.get_x(h_input)
    g4 = net_ufrl4.get_x(h_input)
    g5 = net_ufrl5.get_x(h_input)
    g6 = net_ufrl6.get_x(h_input)

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())

    num_samples = X1.shape[0]
    num_batchs = math.ceil(num_samples / batch_size)

    U1 = X1
    U2 = X2
    U3 = X3
    U4 = X4
    U5 = X5
    U6 = X6
    for j in range(epochs[0]):
        X1, X2, X3, X4, X5, X6, U1, U2, U3, U4, U5, U6, H, gt = shuffle(X1, X2, X3, X4, X5, X6,
                                                                        U1, U2, U3, U4, U5, U6, H, gt)

        print('Updating parameters of UFRL networks and multi-view representation H!')
        for num_batch_i in range(int(num_batchs) - 1):
            start_idx, end_idx = num_batch_i * batch_size, (num_batch_i + 1) * batch_size
            end_idx = min(num_samples, end_idx)
            batch_u1 = U1[start_idx: end_idx, ...]
            batch_u2 = U2[start_idx: end_idx, ...]
            batch_u3 = U3[start_idx: end_idx, ...]
            batch_u4 = U4[start_idx: end_idx, ...]
            batch_u5 = U5[start_idx: end_idx, ...]
            batch_u6 = U6[start_idx: end_idx, ...]
            batch_h = H[start_idx: end_idx, ...]

            sess.run(tf.assign(h_input, batch_h))
            
            _, val_ufrl = sess.run([update_ufrl, loss_ufrl_mse], feed_dict={fea1_latent: batch_u1,
                                                                  fea2_latent: batch_u2,
                                                                  fea3_latent: batch_u3,
                                                                  fea4_latent: batch_u4,
                                                                  fea5_latent: batch_u5,
                                                                  fea6_latent: batch_u6})

            for k in range(epochs[1]):
                sess.run(update_h, feed_dict={fea1_latent: batch_u1, fea2_latent: batch_u2, fea3_latent: batch_u3,
                                              fea4_latent: batch_u4, fea5_latent: batch_u5, fea6_latent: batch_u6})

            batch_h_new = sess.run(h_input)
            H[start_idx: end_idx, ...] = batch_h_new

            sess.run(tf.assign(h_input, batch_h_new))
            batch_g1_new = sess.run(g1, feed_dict={h_input: batch_h})
            batch_g2_new = sess.run(g2, feed_dict={h_input: batch_h})
            batch_g3_new = sess.run(g3, feed_dict={h_input: batch_h})
            batch_g4_new = sess.run(g4, feed_dict={h_input: batch_h})
            batch_g5_new = sess.run(g5, feed_dict={h_input: batch_h})
            batch_g6_new = sess.run(g6, feed_dict={h_input: batch_h})
            G1[start_idx: end_idx, ...] = batch_g1_new
            G2[start_idx: end_idx, ...] = batch_g2_new
            G3[start_idx: end_idx, ...] = batch_g3_new
            G4[start_idx: end_idx, ...] = batch_g4_new
            G5[start_idx: end_idx, ...] = batch_g5_new
            G6[start_idx: end_idx, ...] = batch_g6_new

            output = "\t Epoch : {:.0f} -- Batch : {:.0f} ===> Training loss of UFRL networks = {:.4f}  ".format((j + 1),(num_batch_i + 1), val_ufrl)
            print(output)

        print('Updating view-specific feature representations in multiple views!')

        U1_VFRL = VFRL(k=k_nn_vfrl, measure='cosine')
        U2_VFRL = VFRL(k=k_nn_vfrl, measure='cosine')
        U3_VFRL = VFRL(k=k_nn_vfrl, measure='cosine')
        U4_VFRL = VFRL(k=k_nn_vfrl, measure='cosine')
        U5_VFRL = VFRL(k=k_nn_vfrl, measure='cosine')
        U6_VFRL = VFRL(k=k_nn_vfrl, measure='cosine')
        print('\t Updating the 1-st view!')
        U1 = U1_VFRL.fit(X1, para_beta, G1)
        print('\t Updating the 2-nd view!')
        U2 = U2_VFRL.fit(X2, para_beta, G2)
        print('\t Updating the 3-rd view!')
        U3 = U3_VFRL.fit(X3, para_beta, G3)
        print('\t Updating the 4-th view!')
        U4 = U4_VFRL.fit(X4, para_beta, G4)
        print('\t Updating the 5-th view!')
        U5 = U5_VFRL.fit(X5, para_beta, G5)
        print('\t Updating the 6-th view!')
        U6 = U6_VFRL.fit(X6, para_beta, G6)

    return H, gt


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)