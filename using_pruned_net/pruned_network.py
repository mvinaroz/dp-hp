# To test a pruned network
# A stand-alone script
# (1) we define a model that matches the pruned network
# (2) we load a pruned network to which we set the model parameters
# (3) we evaluate the test accuracy on the pruned model to make sure we loaded the right model with right accuracy

from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import os
import sys
import socket
import numpy as np
import torch.nn.functional as f
import torch
import torch.nn as nn
from VGG_model import VGG

""" (1) load the model and pruned network """
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model2load = 'ckpt_vgg16_prunedto_39,39,63,455,98,97,52,62,22,42,47,47,42,62_90.69.t7'
# model2load = 'ckpt_vgg16_94.34.t7' # this is pre-trained VGG with CIFAR10 data
print('==> Building model..')
net = VGG('VGG15')
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)

checkpoint = torch.load(model2load)
net.load_state_dict(checkpoint['net'], strict=False)

""" (2) the next thing to try out: check if the test accuracy on CIFAR10 is about 90 percent """
###  test data loading ###
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# if you use it first time, make sure to put "download=True" below.
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

net.eval()
test_loss = 0
correct = 0
total = 0
criterion = nn.CrossEntropyLoss()
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, selected_features = net(inputs) # size(selected_features) = minibatch by 47
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

print('Test Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
        test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
print(100.0 * float(correct) / total)
