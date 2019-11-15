"""" test a simple generating training using MMD for relatively simple datasets """
# Mijung wrote on Nov 6, 2019

import numpy as np
import matplotlib.pyplot as plt

# generate data from 2D Gaussian for sanity check
def generate_data(mean_param, cov_param, n):

    how_many_Gaussians = mean_param.shape[1]
    dim_Gaussians = mean_param.shape[0]
    data_samps = np.zeros((n, dim_Gaussians))

    for i in np.arange(0,how_many_Gaussians):
        print(i)

        how_many_samps = np.int(n/how_many_Gaussians)
        new_samps = np.random.multivariate_normal(mean_param[:, i], cov_param[:, :, i], how_many_samps)
        data_samps[(i*how_many_samps):((i+1)*how_many_samps),:] = new_samps
        print((i*how_many_samps))
        print(((i+1)*how_many_samps))
        
    return data_samps


""" use the same generator as in DP-GAN and PATE-GAN """
# this is the lines of code from PATE-GAN paper
# def generator(z, y):
#     inputs = tf.concat([z, y], axis=1)
#     G_h1 = tf.nn.tanh(tf.matmul(inputs, G_W1) + G_b1)
#     G_h2 = tf.nn.tanh(tf.matmul(G_h1, G_W2) + G_b2)
#     G_log_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
#
#     return G_log_prob


# this is the lines of code from DP-GAN paper
# class Generator(object):
#     def __init__(self):
#         self.z_dim = 100
#         self.x_dim = 784
#         self.name = 'mnist/mlp/g_net'
#
#     def __call__(self, z):
#         with tf.variable_scope(self.name) as vs:
#             fc = z
#             fc = tcl.fully_connected(
#                 fc, 512,
#                 weights_initializer=tf.random_normal_initializer(stddev=0.02),
#                 weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
#                 activation_fn=tcl.batch_norm
#             )
#             fc = leaky_relu(fc)
#             fc = tcl.fully_connected(
#                 fc, 512,
#                 weights_initializer=tf.random_normal_initializer(stddev=0.02),
#                 weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
#                 activation_fn=tcl.batch_norm
#             )
#             fc = leaky_relu(fc)
#             fc = tcl.fully_connected(
#                 fc, 512,
#                 weights_initializer=tf.random_normal_initializer(stddev=0.02),
#                 weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
#                 activation_fn=tcl.batch_norm
#             )
#             fc = leaky_relu(fc)
#             fc = tc.layers.fully_connected(
#                 fc, 784,
#                 weights_initializer=tf.random_normal_initializer(stddev=0.02),
#                 weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
#                 activation_fn=tf.sigmoid
#             )
#             return fc

# class Model(nn.Module):
#     #I'm going to define my own Model here following how I generated this dataset
#
#     def __init__(self, input_dim, hidden_dim, W1, b_1, W2, b_2):
#     # def __init__(self, input_dim, hidden_dim):
#         super(Model, self).__init__()
#
#         self.W1 = W1
#         self.b1 = b_1
#         self.W2 = W2
#         self.b2 = b_2
#         self.parameter = Parameter(-1e-10*torch.ones(hidden_dim),requires_grad=True) # this parameter lies
#
#     def forward(self, x):
#
#         pre_activation = torch.mm(x, self.W1)
#         shifted_pre_activation = pre_activation - self.b1
#         phi = F.softplus(self.parameter)
#
#         """directly use mean of Dir RV."""
#         S = phi/torch.sum(phi)
#
#         x = shifted_pre_activation * S
#         x = F.relu(x)
#         x = torch.mm(x, self.W2) + self.b2
#         label = torch.sigmoid(x)
#
#         return label,S


def main():

    n = 1200 # number of data points divisable by num_Gassians
    num_Gaussians = 3
    input_dim = 2
    mean_param = np.zeros((input_dim, num_Gaussians))
    cov_param = np.zeros((input_dim, input_dim, num_Gaussians))

    mean_param[:, 0] = [6, 2]
    mean_param[:, 1] = [-1, 2]
    mean_param[:, 2] = [4, -3]

    cov_param[:, :, 0] = 0.5 * np.eye(input_dim)
    cov_param[:, :, 1] = 1 * np.eye(input_dim)
    cov_param[:, :, 2] = 0.2 * np.eye(input_dim)

    data_samps = generate_data(mean_param, cov_param, n)
    # print(data_samps)
    plt.plot(data_samps[:,0], data_samps[:,1], 'o')
    plt.show()



if __name__ == '__main__':
    main()

