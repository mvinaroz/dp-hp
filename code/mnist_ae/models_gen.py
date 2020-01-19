import torch as pt
import torch.nn as nn


class FCGen(nn.Module):
  def __init__(self, d_code, d_hid, d_enc):
    super(FCGen, self).__init__()
    self.fc1 = nn.Linear(d_code, d_hid)
    self.fc2 = nn.Linear(d_hid, d_hid)
    self.fc3 = nn.Linear(d_hid, d_enc)
    self.relu = nn.ReLU()
    self.d_code = d_code

  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.fc3(x)
    return x

  def get_code(self, batch_size, device):
    return pt.randn(batch_size, self.d_code, device=device)


class FCLabelGen(nn.Module):
  def __init__(self, d_code, d_hid, d_enc, n_labels=10):
    super(FCLabelGen, self).__init__()
    self.fc1 = nn.Linear(d_code, d_hid)
    self.fc2 = nn.Linear(d_hid, d_hid)
    self.data_layer = nn.Linear(d_hid, d_enc)
    self.label_layer = nn.Linear(d_hid, n_labels)
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax(dim=1)
    self.d_code = d_code

  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x_data = self.data_layer(x)
    x_labels = self.softmax(self.label_layer(x))
    return x_data, x_labels

  def get_code(self, batch_size, device):
    return pt.randn(batch_size, self.d_code, device=device)


class FCCondGen(nn.Module):
  def __init__(self, d_code, d_hid, d_enc, n_labels):
    super(FCCondGen, self).__init__()
    self.fc1 = nn.Linear(d_code + n_labels, d_hid)
    self.fc2 = nn.Linear(d_hid, d_hid)
    self.fc3 = nn.Linear(d_hid, d_enc)
    self.relu = nn.ReLU()
    self.d_code = d_code
    self.n_labels = n_labels

  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.fc3(x)
    return x

  def get_code(self, batch_size, device, return_labels=True):
    code = pt.randn(batch_size, self.d_code, device=device)
    sampled_labels = pt.randint(self.n_labels, (batch_size, 1), device=device)
    gen_one_hots = pt.zeros(batch_size, self.n_labels, device=device)
    gen_one_hots.scatter_(1, sampled_labels, 1)
    code = pt.cat([code, gen_one_hots.to(pt.float32)], dim=1)
    # print(code.shape)
    if return_labels:
      return code, gen_one_hots
    else:
      return code



# class FCGen(nn.Module):
#   def __init__(self, d_code, d_hid, d_enc):
#     super(FCGen, self).__init__()
#     layer_spec = [d_code] + list(d_hid) + [d_enc]
#     self.fc_layers = [nn.Linear(layer_spec[k], layer_spec[k + 1]) for k in range(len(layer_spec) - 1)]
#     self.relu = nn.ReLU()
#     # self.tanh = nn.Tanh()
#     self.d_code = d_code
#
#   def forward(self, x):
#     for layer in self.fc_layers[:-1]:
#       x = self.relu(layer(x))
#     x = self.fc_layers[-1](x)
#     return x
#
#   def get_code(self, batch_size, device):
#     return pt.randn(batch_size, self.d_code, device=device)


# class FCLabelGen(nn.Module):
#   def __init__(self, d_code, d_hid, d_enc, n_labels=10):
#     super(FCLabelGen, self).__init__()
#     layer_spec = [d_code] + list(d_hid)
#     self.fc_layers = [nn.Linear(layer_spec[k], layer_spec[k + 1]) for k in range(len(layer_spec) - 1)]
#     self.data_layer = nn.Linear(layer_spec[-1], d_enc)
#     self.label_layer = nn.Linear(layer_spec[-1], n_labels)
#     self.relu = nn.ReLU()
#     self.softmax = nn.Softmax(dim=1)
#     self.d_code = d_code
#
#   def forward(self, x):
#     for layer in self.fc_layers:
#       x = self.relu(layer(x))
#     x_data = self.data_layer(x)
#     x_labels = self.softmax(self.label_layer(x))
#     return x_data, x_labels
#
#   def get_code(self, batch_size, device):
#     return pt.randn(batch_size, self.d_code, device=device)
#
#
# class FCCondGen(nn.Module):
#   def __init__(self, d_code, d_hid, d_enc, n_labels):
#     super(FCCondGen, self).__init__()
#     layer_spec = [d_code + n_labels] + list(d_hid) + [d_enc]
#     self.fc_layers = [nn.Linear(layer_spec[k], layer_spec[k + 1]) for k in range(len(layer_spec) - 1)]
#     self.relu = nn.ReLU()
#     self.d_code = d_code
#     self.n_labels = n_labels
#
#   def forward(self, x):
#     for layer in self.fc_layers[:-1]:
#       x = self.relu(layer(x))
#     x = self.fc_layers[-1](x)
#     return x
#
#   def get_code(self, batch_size, device, return_labels=True):
#     code = pt.randn(batch_size, self.d_code, device=device)
#     sampled_labels = pt.randint(self.n_labels, (batch_size, 1), device=device)
#     gen_one_hots = pt.zeros(batch_size, self.n_labels, device=device)
#     gen_one_hots.scatter_(1, sampled_labels, 1)
#     code = pt.cat([code, gen_one_hots.to(pt.float32)], dim=1)
#     # print(code.shape)
#     if return_labels:
#       return code, gen_one_hots
#     else:
#       return code
