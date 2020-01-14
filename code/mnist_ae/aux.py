import numpy as np
import torch as pt
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')  # to plot without Xserver
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def rff_gauss(x, w):
  """ this is a Pytorch version of Wittawat's code for RFFKGauss"""
  # Fourier transform formula from
  # http://mathworld.wolfram.com/FourierTransformGaussian.html

  xwt = pt.mm(x, w.t())
  z_1 = pt.cos(xwt)
  z_2 = pt.sin(xwt)

  z = pt.cat((z_1, z_2), 1) / pt.sqrt(pt.tensor(w.shape[1]).to(pt.float32))  # w.shape[1] == n_features / 2
  return z


def get_mnist_dataloaders(args, use_cuda):
  mnist_mean = 0.1307
  mnist_sdev = 0.3081
  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
  prep_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mnist_mean,), (mnist_sdev,))])
  train_loader = pt.utils.data.DataLoader(datasets.MNIST('../../data', train=True, transform=prep_transforms),
                                          batch_size=args.batch_size, shuffle=True, **kwargs)
  test_loader = pt.utils.data.DataLoader(datasets.MNIST('../../data', train=False, transform=prep_transforms),
                                         batch_size=args.test_batch_size, shuffle=True, **kwargs)
  return train_loader, test_loader


def plot_mnist_batch(mnist_mat, n_rows, n_cols, save_path):
  n_imgs = n_rows * n_cols
  bs = mnist_mat.shape[0]
  n_to_fill = n_imgs - bs
  mnist_mat = np.reshape(mnist_mat, (bs, 28, 28))
  fill_mat = np.zeros((n_to_fill, 28, 28))
  mnist_mat = np.concatenate([mnist_mat, fill_mat])
  mnist_mat_as_list = [np.split(mnist_mat[n_rows*i:n_rows*(i+1)], n_rows) for i in range(n_cols)]
  # print([k.shape for k in mnist_mat_as_list[0]])
  mnist_mat_flat = np.concatenate([np.concatenate(k, axis=1).squeeze() for k in mnist_mat_as_list], axis=1)
  # print(mnist_mat_flat.shape)
  # print(np.max(mnist_mat_flat), np.min(mnist_mat_flat))
  mnist_mat_flat = denormalize(mnist_mat_flat)
  # print(mnist_mat_flat.shape)
  # print(np.max(mnist_mat_flat), np.min(mnist_mat_flat))
  # print(mnist_mat_flat.dtype)
  save_img(save_path, mnist_mat_flat)


def denormalize(mnist_mat):
  mnist_mean = 0.1307
  mnist_sdev = 0.3081
  return np.clip(mnist_mat * mnist_sdev + mnist_mean, a_min=0., a_max=1.)


def save_img(save_file, img):
  plt.imsave(save_file, img, cmap=cm.gray, vmin=0., vmax=1.)
