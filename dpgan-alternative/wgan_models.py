import torch.nn as nn
import torch as pt
import numpy as np


class Generator(nn.Module):
  def __init__(self, latent_dim, img_shape):
    super(Generator, self).__init__()
    self.img_shape = img_shape
    self.latent_dim = latent_dim

    def block(in_feat, out_feat, normalize=True):
      layers = [nn.Linear(in_feat, out_feat)]
      if normalize:
        layers.append(nn.BatchNorm1d(out_feat, 0.8))
      layers.append(nn.LeakyReLU(0.2, inplace=True))
      return layers

    self.model = nn.Sequential(
      *block(latent_dim, 128, normalize=False),
      *block(128, 256),
      *block(256, 512),
      *block(512, 1024),
      nn.Linear(1024, int(np.prod(img_shape))),
      nn.Tanh()
    )

  def forward(self, z):
    img = self.model(z)
    img = img.view(img.shape[0], *self.img_shape)
    return img

  def get_noise(self, batch_size, device):
    return pt.randn(batch_size, self.latent_dim, device=device)


# class Discriminator(nn.Module):
#   def __init__(self, img_shape):
#     super(Discriminator, self).__init__()
#
#     self.model = nn.Sequential(
#       nn.Linear(int(np.prod(img_shape)), 512),
#       nn.LeakyReLU(0.2, inplace=True),
#       nn.Linear(512, 256),
#       nn.LeakyReLU(0.2, inplace=True),
#       nn.Linear(256, 1),
#     )
#
#   def forward(self, img):
#     img_flat = img.view(img.shape[0], -1)
#     validity = self.model(img_flat)
#     return validity


class Discriminator(nn.Module):
  def __init__(self, img_shape):
    super(Discriminator, self).__init__()
    self.fc1 = nn.Linear(int(np.prod(img_shape)), 512)
    self.act1 = nn.LeakyReLU(0.2, inplace=True)
    self.fc2 = nn.Linear(512, 256)
    self.act2 = nn.LeakyReLU(0.2, inplace=True)
    self.fc3 = nn.Linear(256, 1)

  def forward(self, img):
    img_flat = img.view(img.shape[0], -1)
    x = self.act1(self.fc1(img_flat))
    x = self.act2(self.fc2(x))
    x = self.fc3(x)
    return x
