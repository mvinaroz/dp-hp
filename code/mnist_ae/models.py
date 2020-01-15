import torch as pt
import torch.nn as nn
#  BASIC CONVOLUTIONAL AUTOENCODER:


class ConvEncoder(nn.Module):

  def __init__(self, d_enc, nc=(1, 2, 4, 4), extra_conv=False):
    super(ConvEncoder, self).__init__()
    # nc = (1, 4, 8, 16)  # n channels
    self.conv1 = nn.Conv2d(nc[0], nc[1], kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(nc[1], nc[2], kernel_size=4, stride=2, padding=1)  # down to 14x14
    self.conv3 = nn.Conv2d(nc[2], nc[2], kernel_size=3, stride=1, padding=1) if extra_conv else None
    self.conv4 = nn.Conv2d(nc[2], nc[3], kernel_size=4, stride=2, padding=1)  # down to 7x7
    self.fc = nn.Linear(7*7*nc[3], d_enc)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.relu(self.conv1(x))
    x = self.relu(self.conv2(x))
    x = self.relu(self.conv3(x)) if self.conv3 is not None else x
    # print(x.shape)
    x = self.relu(self.conv4(x))
    # print(x.shape)
    x = x.reshape(x.shape[0], -1)
    # print(x.shape)
    x = self.fc(x)
    return x


class ConvDecoder(nn.Module):

  def __init__(self, d_enc, nc=(4, 4, 2, 1), extra_conv=False):
    super(ConvDecoder, self).__init__()
    self.fc = nn.Linear(d_enc, 7*7*nc[0])
    self.conv1 = nn.Conv2d(nc[0], nc[1], kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.ConvTranspose2d(nc[1], nc[2], kernel_size=2, stride=2)  # up to 14x14
    self.conv3 = nn.Conv2d(nc[2], nc[2], kernel_size=3, stride=1, padding=1) if extra_conv else None
    self.conv4 = nn.ConvTranspose2d(nc[2], nc[3], kernel_size=2, stride=2)  # up to 28x28
    self.relu = nn.ReLU()
    self.tanh = nn.Tanh()

    self.nc = nc

  def forward(self, x):
    x = self.relu(self.fc(x))
    # print(x.shape)
    x = x.reshape(x.shape[0], self.nc[0], 7, 7)
    # print(x.shape)
    x = self.relu(self.conv1(x))
    # print(x.shape)
    x = self.relu(self.conv2(x))
    # print(x.shape)
    x = self.relu(self.conv3(x)) if self.conv3 is not None else x
    # print(x.shape)
    x = self.conv4(x)
    # print(x.shape)
    return x


class FCEncoder(nn.Module):

  def __init__(self, d_enc, d_hid=(300, 100), extra_layer=False):
    super(FCEncoder, self).__init__()
    # nc = (1, 4, 8, 16)  # n channels
    self.fc1 = nn.Linear(28*28, d_hid[0])
    if extra_layer:
      self.fc2 = nn.Linear(d_hid[0], d_hid[1])
      self.fc3 = nn.Linear(d_hid[1], d_enc)
    else:
      self.fc2 = None
      self.fc3 = nn.Linear(d_hid[0], d_enc)

    self.relu = nn.ReLU()

  def forward(self, x):
    x = x.reshape(x.shape[0], -1)
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x)) if self.fc2 is not None else x
    x = self.fc3(x)
    return x


class FCDecoder(nn.Module):

  def __init__(self, d_enc, d_hid=(100, 300), extra_layer=False, do_reshape=True):
    super(FCDecoder, self).__init__()
    self.fc1 = nn.Linear(d_enc, d_hid[0])

    if extra_layer:
      self.fc2 = nn.Linear(d_hid[0], d_hid[1])
      self.fc3 = nn.Linear(d_hid[1], 28*28)
    else:
      self.fc2 = None
      self.fc3 = nn.Linear(d_hid[0], 28*28)
    self.relu = nn.ReLU()
    self.tanh = nn.Tanh()
    self.do_reshape = do_reshape

  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x)) if self.fc2 is not None else x
    x = self.fc3(x)
    # print(x.shape)
    x = x.reshape(x.shape[0], 1, 28, 28) if self.do_reshape else x
    # print(x.shape)
    return x


class FCGenerator(nn.Module):
  def __init__(self, d_code, d_hid=100, extra_layer=False):
    super(FCGenerator, self).__init__()
    self.fc1 = nn.Linear(d_code, d_hid)
    self.fc2 = nn.Linear(d_hid, d_hid) if extra_layer else None
    self.fc3 = nn.Linear(d_hid, 28*28)
    self.relu = nn.ReLU()
    # self.tanh = nn.Tanh()
    self.d_code = d_code

  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x)) if self.fc2 is not None else x
    # x = self.tanh(self.fc3(x))
    x = self.fc3(x)
    x = x.reshape(x.shape[0], 1, 28, 28)
    return x

  def get_code(self, batch_size):
    return pt.randn(batch_size, self.d_code)


class FCLatentGenerator(nn.Module):
  def __init__(self, d_code, d_enc, d_hid=100, extra_layer=False):
    super(FCLatentGenerator, self).__init__()
    self.fc1 = nn.Linear(d_code, d_hid)
    self.fc2 = nn.Linear(d_hid, d_hid) if extra_layer else None
    self.fc3 = nn.Linear(d_hid, d_enc)
    self.relu = nn.ReLU()
    # self.tanh = nn.Tanh()
    self.d_code = d_code

  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x)) if self.fc2 is not None else x
    # x = self.tanh(self.fc3(x))
    x = self.fc3(x)
    return x

  def get_code(self, batch_size):
    return pt.randn(batch_size, self.d_code)
