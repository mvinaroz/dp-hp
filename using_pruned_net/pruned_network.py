# To test a pruned network
# A stand-alone script
# (1) we define a model that matches the pruned network
# (2) we load a pruned network to which we set the model parameters
# (3) we evaluate the test accuracy on the pruned model
# (4) we then make a forward pass of each data in MNIST to create a smaller dimensional dataset.


from __future__ import print_function
import torch
import torch.optim as optim
import os
import sys
import socket
import numpy as np
import torch.nn.functional as f
import torch
import torch.nn as nn
from VGG16_model import VGG


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model2load = 'ckpt_vgg16_prunedto_39,39,63,455,98,97,52,62,22,42,47,47,42,62_90.69.t7'
print('==> Building model..')
net = VGG('VGG16') #watch
checkpoint = torch.load(model2load)
net.load_state_dict(checkpoint['net'], strict=False)
# print(net.module.c1.weight)
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']


