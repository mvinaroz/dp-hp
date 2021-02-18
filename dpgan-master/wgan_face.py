import os
import time
import argparse
import importlib
import tensorflow as tf
import tensorflow.contrib as tc
import matplotlib
matplotlib.use('Agg')
import cPickle as pickle
from numpy import linalg, argmin, array, arange
import matplotlib.gridspec as gridspec
from utilize import loaddata_face
import logging # these 2 lines ar used in GPU3
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from visualize import *


class WassersteinGAN(object):
    def __init__(self, g_net, d_net, x_sampler, z_sampler, data, model, batch_size=64): # changed
        self.model = model
        self.data = data
        self.g_net = g_net
        self.d_net = d_net
        self.x_sampler = x_sampler
        self.z_sampler = z_sampler
        self.x_dim = self.d_net.x_dim
        self.z_dim = self.g_net.z_dim
        # self.x = tf.placeholder(tf.float32, [None] + self.x_dim, name='x')
        # self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.x = tf.placeholder(tf.float32, [None, self.x_dim], name='x')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        self.x_ = self.g_net(self.z)

        self.d = self.d_net(self.x, reuse=False)
        self.d_ = self.d_net(self.x_)

        self.g_loss = tf.reduce_mean(self.d_)
        self.d_loss = tf.reduce_mean(self.d) - tf.reduce_mean(self.d_)

        self.reg = tc.layers.apply_regularization(
            tc.layers.l1_regularizer(2.5e-5),
            weights_list=[var for var in tf.global_variables() if 'weights' in var.name]
        )
        self.g_loss_reg = self.g_loss + self.reg
        self.d_loss_reg = self.d_loss + self.reg

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_rmsprop = tf.train.RMSPropOptimizer(learning_rate=5e-5)  # DP case
            grads_and_vars = self.d_rmsprop.compute_gradients(-1*self.d_loss_reg, var_list=self.d_net.vars)
            dp_grads_and_vars = []  # noisy version
            for gv in grads_and_vars:  # for each pair
                g = gv[0]  # get the gradient, type in loop one: Tensor("gradients/AddN_37:0", shape=(4, 4, 1, 64), dtype=float32)
                #print g # shape of all vars
                if g is not None:  # skip None case
                    g = self.dpnoise(g, batch_size)  # add noise on the tensor, type in loop one: Tensor("Add:0", shape=(4, 4, 1, 64), dtype=float32)
                dp_grads_and_vars.append((g, gv[1]))
            self.d_rmsprop_new = self.d_rmsprop.apply_gradients(dp_grads_and_vars) # should assign to a new optimizer
            # self.d_rmsprop = tf.train.RMSPropOptimizer(learning_rate=5e-5) \
            #     .minimize(-1*self.d_loss_reg, var_list=self.d_net.vars) # non-DP case
            self.g_rmsprop = tf.train.RMSPropOptimizer(learning_rate=5e-5) \
                .minimize(-1*self.g_loss_reg, var_list=self.g_net.vars)

        self.d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.d_net.vars]
        self.d_net_var_grad = [i for i in tf.gradients(self.d_loss_reg, self.d_net.vars) if i is not None] # explore the effect of noise on norm of D net variables's gradient vector, also remove None type
        self.norm_d_net_var_grad = []
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))
        self.g_loss_store = []  # store loss of generator
        self.d_loss_store = [] # store loss of discriminator
        self.wdis_store = []  # store Wasserstein distance, new added
        # print self.d_net.vars
        # print self.g_net.vars

    def train(self, batch_size=64, num_batches=500000):
        plt.ion()
        self.sess.run(tf.initialize_all_variables())
        start_time = time.time()
        for t in range(0, num_batches):
            d_iters = 5
            if t % 500 == 0 or t < 25: # make the discriminator more accurate at certain iterations
                 d_iters = 100

            for _ in range(0, d_iters): # train discriminator
                data_td = self.x_sampler(batch_size) # data_td: data for training discriminator, data_td.shape: (64, 32, 32, 1)
                bz = self.z_sampler(batch_size, self.z_dim)
                self.sess.run(self.d_clip)
                self.sess.run(self.d_rmsprop_new, feed_dict={self.x: data_td, self.z: bz}) # DP case
                # self.sess.run(self.d_rmsprop, feed_dict={self.x: data_td, self.z: bz}) # non-DP case

            bz = self.z_sampler(batch_size, self.z_dim) # train generator, another batch of z sample
            self.sess.run(self.g_rmsprop, feed_dict={self.z: bz, self.x: data_td})

            if t % 100 == 0: # evaluate loss and norm of gradient vector
                bx = self.x_sampler(batch_size) # the reason we generate another batch of sample is that we want to see if the distance of 2 distributions are indeed pulled closer
                bz = self.z_sampler(batch_size, self.z_dim)

                rd_loss = self.sess.run(
                    self.d_loss, feed_dict={self.x: bx, self.z: bz}
                )
                rg_loss = self.sess.run(
                    self.g_loss, feed_dict={self.z: bz, self.x: bx}
                )
                # d_net_var_grad_val = self.sess.run(self.d_net_var_grad, feed_dict={self.x: bx, self.z: bz})
                # if type(d_net_var_grad_val) != type([0]):
                #     d_net_var_grad_val = [d_net_var_grad_val]
                # self.norm_d_net_var_grad.append(self.norm_w(d_net_var_grad_val))
                print('Iter [%8d] Time [%5.4f] d_loss [%.4f] g_loss [%.4f]' %
                        (t, time.time() - start_time, rd_loss, rg_loss))

                # store rd_loss, rg_loss and W-dis, new added
                self.g_loss_store.append(rg_loss)  # g_loss will increase, here is not self.g_loss nor self.g_loss_reg
                self.d_loss_store.append(rd_loss)  # d_loss will decrease
                self.wdis_store.append(rd_loss)  # Wasserstein distance will decrease

            if t % 1000 == 0: # generate image
                bz = self.z_sampler(1, self.z_dim) # changed, only generate 1 image
                bx = self.sess.run(self.x_, feed_dict={self.z: bz})
                bx = xs.data2img(bx)
                fig = plt.figure(self.data + '.' + self.model)
                grid_show(fig, bx, xs.shape)
                fig.savefig('./result/genefig/{}/{}.jpg'.format(self.data, t)) # changed

            if t % 100000 == 0:  # store generator and discriminator, new added
                saver = tf.train.Saver()
                save_path = saver.save(self.sess, "result/sesssave/sess.ckpt")
                print("Session saved in file: %s" % save_path)

        N = 10 # generate images from generator, after finish training
        z_sample = self.z_sampler(N, self.z_dim)
        x_gene = self.sess.run(self.x_, feed_dict={self.z: z_sample})
        path = './face/CelebA/img_align_celeba_10000_1st_r_28/'
        face_data = loaddata_face(path, len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]))
        face_data_n = array(face_data) # face_data is already normlized (/255)
        x_training_data = []  # corresponding nearest training points in whole face data
        for i in range(N):
            x_ind = self.find(x_gene[i], face_data_n) # find the nearest training point for each generated data point in whole face data
            x_training_data.append(face_data_n[x_ind])

        x_gene = x_gene.tolist() # all to list type
        x_training_data = [i.tolist() for i in x_training_data]

        # store generated data, nearest data (label) and figures
        with open('./result/genefinalfig/x_gene.pickle', 'wb') as fp:
            pickle.dump(x_gene, fp)
        with open('./result/genefinalfig/x_training_data.pickle', 'wb') as fp:
            pickle.dump(x_training_data, fp)
        with open('./result/genefinalfig/norm_d_net_var_grad.pickle', 'wb') as fp:
            pickle.dump(self.norm_d_net_var_grad, fp)
        x_gene = array(x_gene)*255 # to 0-255 scale, rbg image
        x_training_data = array(x_training_data)*255
        plt.figure(figsize=(5, 60))
        G = gridspec.GridSpec(N, 1)
        for i in range(N):
            plt.subplot(G[i, :])
            plt.imshow(x_gene[i], interpolation='nearest')
            plt.xticks(())
            plt.yticks(())
        plt.tight_layout()
        plt.savefig('./result/genefinalfig/x_gene.png')
        plt.clf()
        for i in range(N):
            plt.subplot(G[i, :])
            plt.imshow(x_training_data[i], interpolation='nearest')
            plt.xticks(())
            plt.yticks(())
        plt.tight_layout()
        plt.savefig('./result/genefinalfig/x_training_data.png')

        # store generator and discriminator
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, "result/sesssave/sess.ckpt")
        print("Training finished, session saved in file: %s" % save_path)

    def dpnoise(self, tensor, batch_size):
        '''add noise to tensor'''
        s = tensor.get_shape().as_list()  # get shape of the tensor
        sigma = 0.00001  # assign it manually
        cg = 160000.0
        rt = tf.random_normal(s, mean=0.0, stddev=sigma * cg)
        t = tf.add(tensor, tf.scalar_mul((1.0 / batch_size), rt))
        return t

    def loss_store(self):
        '''store everything new added'''
        # store figure
        t = arange(len(self.g_loss_store))
        plt.close() # clears the entire current figure with all its axes
        plt.plot(t, self.g_loss_store, 'b--')
        plt.xlabel('Generator iterations (*10^{2})')
        plt.ylabel('Generator loss')
        plt.savefig('./result/lossfig/gloss.jpg')
        plt.clf()
        plt.plot(t, self.d_loss_store, 'b--')
        plt.xlabel('Generator iterations (*10^{2})')
        plt.ylabel('Discriminator loss')
        plt.savefig('./result/lossfig/dloss.jpg')
        plt.clf()
        plt.plot(t, self.wdis_store, 'b--')
        plt.xlabel('Generator iterations (*10^{2})')
        plt.ylabel('Wasserstein distance')
        plt.savefig('./result/lossfig/wdis.jpg')
        plt.clf()
        plt.plot(t, self.norm_d_net_var_grad, 'b--')
        plt.xlabel('Generator iterations (*10^{2})')
        plt.ylabel('Norm of gradient vector')
        plt.savefig('./result/lossfig/ngv.jpg')
        # store to file
        gpick = file("result/lossfile/gloss.pckl", "w")
        pickle.dump(self.g_loss_store, gpick)
        gpick.close()
        dpick = file("result/lossfile/dloss.pckl", "w")
        pickle.dump(self.d_loss_store, dpick)
        dpick.close()
        wpick = file("result/lossfile/wdis.pckl", "w")
        pickle.dump(self.wdis_store, wpick)
        wpick.close()
        npick = file("result/lossfile/ngv.pckl", "w")
        pickle.dump(self.norm_d_net_var_grad, npick)
        npick.close()

    def find(self, gen, train):
        dist = []
        for i in range(len(train)):
            dist.append(linalg.norm(array(gen) - array(train[i])))
        return argmin(dist)

    def norm_w(self, v):
        return sum([linalg.norm(i) for i in v])

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='face')
    parser.add_argument('--model', type=str, default='dcgan')
    parser.add_argument('--gpus', type=str, default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    data = importlib.import_module(args.data) # from parser
    model = importlib.import_module(args.data + '.' + args.model)
    xs = data.DataSampler()
    zs = data.NoiseSampler()
    d_net = model.Discriminator()
    g_net = model.Generator()
    wgan = WassersteinGAN(g_net, d_net, xs, zs, args.data, args.model)
    wgan.train()
    wgan.loss_store() # new added