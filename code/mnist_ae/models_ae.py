import torch as pt
import torch.nn as nn
import backpack

# class ConvEncoder(nn.Module):
#
#   def __init__(self, d_enc, nc=(1, 2, 4, 4), extra_conv=False):
#     super(ConvEncoder, self).__init__()
#     # nc = (1, 4, 8, 16)  # n channels
#     self.conv1 = nn.Conv2d(nc[0], nc[1], kernel_size=3, stride=1, padding=1)
#     self.conv2 = nn.Conv2d(nc[1], nc[2], kernel_size=4, stride=2, padding=1)  # down to 14x14
#     self.conv3 = nn.Conv2d(nc[2], nc[2], kernel_size=3, stride=1, padding=1) if extra_conv else None
#     self.conv4 = nn.Conv2d(nc[2], nc[3], kernel_size=4, stride=2, padding=1)  # down to 7x7
#     self.fc = nn.Linear(7*7*nc[3], d_enc)
#     self.relu = nn.ReLU()
#
#   def forward(self, x):
#     x = self.relu(self.conv1(x))
#     x = self.relu(self.conv2(x))
#     x = self.relu(self.conv3(x)) if self.conv3 is not None else x
#     x = self.relu(self.conv4(x))
#     x = x.reshape(x.shape[0], -1)
#     x = self.fc(x)
#     return x
#
#
# class ConvDecoder(nn.Module):
#
#   def __init__(self, d_enc, nc=(4, 4, 2, 1), extra_conv=False):
#     super(ConvDecoder, self).__init__()
#     self.fc = nn.Linear(d_enc, 7*7*nc[0])
#     self.conv1 = nn.Conv2d(nc[0], nc[1], kernel_size=3, stride=1, padding=1)
#     self.conv2 = nn.ConvTranspose2d(nc[1], nc[2], kernel_size=2, stride=2)  # up to 14x14
#     self.conv3 = nn.Conv2d(nc[2], nc[2], kernel_size=3, stride=1, padding=1) if extra_conv else None
#     self.conv4 = nn.ConvTranspose2d(nc[2], nc[3], kernel_size=2, stride=2)  # up to 28x28
#     self.relu = nn.ReLU()
#     self.tanh = nn.Tanh()
#     self.nc = nc
#
#   def forward(self, x):
#     x = self.relu(self.fc(x))
#     x = x.reshape(x.shape[0], self.nc[0], 7, 7)
#     x = self.relu(self.conv1(x))
#     x = self.relu(self.conv2(x))
#     x = self.relu(self.conv3(x)) if self.conv3 is not None else x
#     x = self.conv4(x)
#     return x


# class FCEncoder(nn.Module):
#
#   def __init__(self, d_enc, d_hid=(300, 100), extra_layer=False):
#     super(FCEncoder, self).__init__()
#     # nc = (1, 4, 8, 16)  # n channels
#     self.fc1 = nn.Linear(28*28, d_hid[0])
#     if extra_layer:
#       self.fc2 = nn.Linear(d_hid[0], d_hid[1])
#       self.fc3 = nn.Linear(d_hid[1], d_enc)
#     else:
#       self.fc2 = None
#       self.fc3 = nn.Linear(d_hid[0], d_enc)
#
#     self.relu = nn.ReLU()
#
#   def forward(self, x):
#     x = x.reshape(x.shape[0], -1)
#     x = self.relu(self.fc1(x))
#     x = self.relu(self.fc2(x)) if self.fc2 is not None else x
#     x = self.fc3(x)
#     return x
#
#
# class FCDecoder(nn.Module):
#
#   def __init__(self, d_enc, d_hid=(100, 300), extra_layer=False, do_reshape=True, use_sigmoid=False):
#     super(FCDecoder, self).__init__()
#     self.fc1 = nn.Linear(d_enc, d_hid[0])
#
#     if extra_layer:
#       self.fc2 = nn.Linear(d_hid[0], d_hid[1])
#       self.fc3 = nn.Linear(d_hid[1], 28*28)
#     else:
#       self.fc2 = None
#       self.fc3 = nn.Linear(d_hid[0], 28*28)
#     self.relu = nn.ReLU()
#     self.sigmoid = nn.Sigmoid()
#     self.do_reshape = do_reshape
#     self.use_sigmoid = use_sigmoid
#
#   def forward(self, x):
#     x = self.relu(self.fc1(x))
#     x = self.relu(self.fc2(x)) if self.fc2 is not None else x
#     x = self.fc3(x)
#     if self.use_sigmoid:
#       x = self.sigmoid(x)
#     x = x.reshape(x.shape[0], 1, 28, 28) if self.do_reshape else x
#     return x


class FCEnc(nn.Module):

  def __init__(self, d_in, d_hid, d_enc, reshape=False):
    super(FCEnc, self).__init__()
    # nc = (1, 4, 8, 16)  # n channels
    layer_spec = [d_in] + list(d_hid) + [d_enc]
    self.fc_layers = [nn.Linear(layer_spec[k], layer_spec[k+1]) for k in range(len(layer_spec) - 1)]
    self.relu = nn.ReLU()
    self.reshape = reshape

  def forward(self, x):
    if self.reshape:
      x = x.reshape(x.shape[0], -1)

    for layer in self.fc_layers[:-1]:
      x = self.relu(layer(x))
    x = self.fc_layers[-1](x)

    return x


class FCDec(nn.Module):

  def __init__(self, d_enc, d_hid, d_out, use_sigmoid=False, reshape=False):
    super(FCDec, self).__init__()
    layer_spec = [d_enc] + list(d_hid) + [d_out]
    self.fc_layers = [nn.Linear(layer_spec[k], layer_spec[k + 1]) for k in range(len(layer_spec) - 1)]
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
    self.use_sigmoid = use_sigmoid
    self.reshape = reshape

  def forward(self, x):
    for layer in self.fc_layers[:-1]:
      x = self.relu(layer(x))
    x = self.fc_layers[-1](x)
    if self.use_sigmoid:
      x = self.sigmoid(x)
    if self.reshape:
      x = x.reshape(x.shape[0], 1, 28, 28)
    return x


class ConvEnc(nn.Module):

  def __init__(self, d_enc, layer_spec):
    super(ConvEnc, self).__init__()
    assert layer_spec[-1][0] == 'f'  # last layer must be linear
    layer_spec[-1][1][1] = d_enc  # set linear output to d_enc
    self.layer_spec = layer_spec
    self.layers = conv_layers_by_spec(self.layer_spec)

  def forward(self, x):
    x = conv_forward_pass_by_spec(x, self.layer_spec, self.layers)
    return x


class ConvDec(nn.Module):

  def __init__(self, d_enc, layer_spec):
    super(ConvDec, self).__init__()
    assert layer_spec[0][0] == 'f'  # first layer must be linear
    layer_spec[0][1][0] = d_enc  # set linear input to d_enc
    self.layer_spec = layer_spec
    self.layers = conv_layers_by_spec(self.layer_spec)

  def forward(self, x):
    x = conv_forward_pass_by_spec(x, self.layer_spec, self.layers)
    if self.use_sigmoid:
      x = self.sigmoid(x)
    return x


def conv_layers_by_spec(spec):
  layers = []
  for key, v in spec:
    if key == 'c':  # format: cin-cout-kernel-stride-padding
      layer = nn.Conv2d(v[0], v[1], v[2], stride=v[3], padding=v[4])
    elif key == 'f':  # format: din-dout
      layer = nn.Linear(v[0], v[1])
    elif key == 'r':  # format: d0-d1-d2-....
      def reshape_fun(x):
        return pt.reshape(x, shape=[-1] + list(v))
      layer = reshape_fun
    elif key == 'u':  # format: scale
      layer = nn.UpsamplingBilinear2d(scale_factor=v[0])
    else:
      raise KeyError
    layers.append(layer)
  return layers


def conv_forward_pass_by_spec(x, spec, layers):
  for idx, (s, l) in enumerate(zip(spec, layers)):
    l(x)
    if (s[0] == 'f' or s[0] == 'c') and idx+1 < len(layers):
      x = nn.functional.relu(x)
  return x


def default_cnn_specs():
  enc = 'c1-2-3-1-1,c2-2-4-2-1,c2-4-3-1-1,c4-4-4-2-1,r'

#     self.conv1 = nn.Conv2d(nc[0], nc[1], kernel_size=3, stride=1, padding=1)
#     self.conv2 = nn.Conv2d(nc[1], nc[2], kernel_size=4, stride=2, padding=1)  # down to 14x14
#     self.conv3 = nn.Conv2d(nc[2], nc[2], kernel_size=3, stride=1, padding=1) if extra_conv else None
#     self.conv4 = nn.Conv2d(nc[2], nc[3], kernel_size=4, stride=2, padding=1)  # down to 7x7
#     self.fc = nn.Linear(7*7*nc[3], d_enc)
#     self.relu = nn.ReLU()
#
#   def forward(self, x):
#     x = self.relu(self.conv1(x))
#     x = self.relu(self.conv2(x))
#     x = self.relu(self.conv3(x)) if self.conv3 is not None else x
#     x = self.relu(self.conv4(x))
#     x = x.reshape(x.shape[0], -1)
#     x = self.fc(x)
#     return x
#
#
# class ConvDecoder(nn.Module):
#
#   def __init__(self, d_enc, nc=(4, 4, 2, 1), extra_conv=False):
#     super(ConvDecoder, self).__init__()
#     self.fc = nn.Linear(d_enc, 7*7*nc[0])
#     self.conv1 = nn.Conv2d(nc[0], nc[1], kernel_size=3, stride=1, padding=1)
#     self.conv2 = nn.ConvTranspose2d(nc[1], nc[2], kernel_size=2, stride=2)  # up to 14x14
#     self.conv3 = nn.Conv2d(nc[2], nc[2], kernel_size=3, stride=1, padding=1) if extra_conv else None
#     self.conv4 = nn.ConvTranspose2d(nc[2], nc[3], kernel_size=2, stride=2)  # up to 28x28
#     self.relu = nn.ReLU()
#     self.tanh = nn.Tanh()
#     self.nc = nc
#
#   def forward(self, x):
#     x = self.relu(self.fc(x))
#     x = x.reshape(x.shape[0], self.nc[0], 7, 7)
#     x = self.relu(self.conv1(x))
#     x = self.relu(self.conv2(x))
#     x = self.relu(self.conv3(x)) if self.conv3 is not None else x
#     x = self.conv4(x)
#     return x