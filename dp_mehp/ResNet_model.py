# code taken from https://github.com/nanekja/JovianML-Project

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


class ResNet(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        # self.classifier = nn.Sequential(nn.MaxPool2d(3),
        #                                 nn.Flatten(),
        #                                 nn.Linear(512, num_classes))

        self.maxpool = nn.MaxPool2d(3)
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(512, num_classes)

    def forward(self, xb):

        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out

        output = self.maxpool(out)
        flattened_output = self.flatten(output)
        output = self.l1(flattened_output)

        # out = self.classifier(out)
        return output, flattened_output



############################ classifier using VGG features #################################

class Classifier_ResNet(nn.Module):

    def __init__(self, features):
        super(Classifier_ResNet, self).__init__()

        self.features = features
        self.parameter = Parameter(torch.zeros(512,10), requires_grad=True)
        # self.l1 = nn.Linear(512,10)

    def forward(self, x):  # x is mini_batch_size by input_dim

        _, x = self.features(x)
        # output = self.l1(x)
        output = torch.matmul(x, self.parameter)

        return output


