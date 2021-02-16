# code taken from https://github.com/nanekja/JovianML-Project
import os
import tarfile
import torch
import numpy as np

import torchvision
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader

import torch.nn as nn
from torch.nn import AvgPool2d
import torch.nn.functional as F

import matplotlib.pyplot as plt
from ResNet_model import ResNet
# from torchsummary import summary

from scipy.io import loadmat

def show(image):
  ig, axes = plt.subplots()
  plt.imshow(image)

def svhn_loader(batch_size):

    root='http://ufldl.stanford.edu/housenumbers/'

    transform = transforms.Compose([
                                    transforms.CenterCrop((28, 28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                                    ])

    train = torchvision.datasets.SVHN(root, split='train', transform=transform, target_transform=None, download=True)

    train_dl = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    test_vi = torchvision.datasets.SVHN(root, split='test', transform=transforms.ToTensor(),download=True)
    test_vii = torch.utils.data.DataLoader(test_vi, batch_size=batch_size, shuffle=True)

    test = torchvision.datasets.SVHN(root, split='test', transform=transform, target_transform=None, download=True)

    test_dl = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)

    images_train, labels_train = next(iter(train_dl))
    images_test, labels_test = next(iter(test_dl))
    images_test_vi, labels_test_vi = next(iter(test_vi))

    print("Shape of train inputs: ", images_train.shape, "; Shape of train labels: ", labels_train.shape)
    print("Shape of test inputs: ",images_test.shape, "; Shape of test inputs: ", labels_test.shape)
    print("Batch size =", batch_size)

    return test_vii, train_dl, test_dl


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')


def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []

    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)

    # save the trained model
    torch.save({'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}, 'Trained_ResNet')

    return history

def main():

    visualize = False

    # Load data
    batch_size = 128
    test_vii, train_dl, test_dl = svhn_loader(batch_size)

    # visualize data if visualize = True
    if visualize:
        dataiter = iter(test_vii)
        images, labels = dataiter.next()
        for i in range(113,115):
          img = images[i]
          img = img.numpy().transpose((1, 2, 0))
          show(img)

        plt.show()

    # define a model
    device = get_default_device()
    print(device)
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(test_dl, device)

    num_epochs = 10
    opt_func = torch.optim.Adam
    lr = 0.001
    num_classes = 10

    Resnet_model = to_device(ResNet(3, num_classes), device)
    print(Resnet_model)

    grad_clip = 0.1
    weight_decay = 1e-4
    history = fit_one_cycle(num_epochs, lr, Resnet_model, train_dl, val_dl, grad_clip=grad_clip, weight_decay=weight_decay, opt_func=opt_func)
    plot_accuracies(history)
    plt.show()


if __name__ == '__main__':
    main()