import numpy as np
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import util
import random
import argparse
import seaborn as sns
sns.set()
# %matplotlib inline
from autodp import privacy_calibrator

import warnings
warnings.filterwarnings('ignore')
import os

def RFF_Gauss(n_features, X, W, device):
  """ this is a Pytorch version of Wittawat's code for RFFKGauss"""

  W = torch.Tensor(W).to(device)
  X = X.to(device)
  XWT = torch.mm(X, torch.t(W)).to(device)
  Z1 = torch.cos(XWT)
  Z2 = torch.sin(XWT)
  Z = torch.cat((Z1, Z2),1) * torch.sqrt(2.0/torch.Tensor([n_features])).to(device)
  return Z

class Generative_Model(nn.Module):
  def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
    super(Generative_Model, self).__init__()

    self.input_size = input_size
    self.hidden_size_1 = hidden_size_1
    self.hidden_size_2 = hidden_size_2
    self.output_size = output_size

    self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
    self.bn1 = torch.nn.BatchNorm1d(self.hidden_size_1)
    self.relu = torch.nn.ReLU()
    self.fc2 = torch.nn.Linear(self.hidden_size_1, self.hidden_size_2)
    self.bn2 = torch.nn.BatchNorm1d(self.hidden_size_2)
    self.fc3 = torch.nn.Linear(self.hidden_size_2, self.output_size)

  def forward(self, x):
    hidden = self.fc1(x)
    relu = self.relu(self.bn1(hidden))
    output = self.fc2(relu)
    output = self.relu(self.bn2(output))
    output = self.fc3(output)

    return output

def main():

    # load data
    data = np.load('mixture_of_Gaussians_N100000_D5.npz')
    data_samps = data.f.X_private

    n, input_dim = data_samps.shape
    mini_batch_size = 4000
    n_features = 10000

    input_size = 10 # input size to the generator
    hidden_size_1 = 4 * input_size
    hidden_size_2 = 2 * input_size
    output_size = input_dim

    model = Generative_Model(input_size=input_size, hidden_size_1=hidden_size_1,
                                              hidden_size_2=hidden_size_2,
                                              output_size=output_size)

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    how_many_epochs = 500
    how_many_iter = np.int(n/mini_batch_size)

    training_loss_per_epoch = np.zeros(how_many_epochs)

    med = util.meddistance(data_samps)
    sigma2 = med**2
    print('length scale from median heuristic is', sigma2)

    draws = n_features // 2
    W_freq =  np.random.randn(draws, input_dim) / np.sqrt(sigma2)
    mean_emb1 = torch.mean(RFF_Gauss(n_features, torch.Tensor(data_samps), W_freq), axis=0)

    print('Starting Training')

    for epoch in range(how_many_epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        for i in range(how_many_iter):

            # zero the parameter gradients
            optimizer.zero_grad()
            input_to_the_generator = torch.randn(mini_batch_size, input_size)
            outputs = model(input_to_the_generator)

            mean_emb2 = torch.mean(RFF_Gauss(n_features, outputs, W_freq), axis=0)

            loss = torch.norm(mean_emb1-mean_emb2, p=2)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        if running_loss<=1e-4:
            break
        print('epoch # and running loss are ', [epoch, running_loss])
        training_loss_per_epoch[epoch] = running_loss



if __name__ == '__main__':
    main()