import os
import torch as pt
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms


def get_dataloader(batch_size):
  # Configure data loader
  os.makedirs("../data", exist_ok=True)
  data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
  dataset = datasets.MNIST("../data", train=True, download=True, transform=data_transform)
  dataloader = pt.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
  return dataloader


def get_single_label_dataloader(batch_size, label_idx):
  os.makedirs("../data", exist_ok=True)
  data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
  dataset = datasets.MNIST("../data", train=True, download=True, transform=data_transform)
  selected_ids = dataset.targets == label_idx
  dataset.data = dataset.data[selected_ids]
  dataset.targets = dataset.targets[selected_ids]
  n_data = dataset.data.shape[0]
  dataloader = pt.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
  return dataloader, n_data


if __name__ == '__main__':
  get_single_label_dataloader(100, 0)
