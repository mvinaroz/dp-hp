# in this script, we will train output layers only for MNIST/FashionMNIST classification
# using the pruned and trained VGG15 with CIFAR10 data

from __future__ import print_function
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as f
import torch
from VGG_model import VGG, Classifier
import matplotlib.pyplot as plt

def load_pruned_model(model2load, device):

    print('==> Building model..')
    net = VGG('VGG15')
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)

    checkpoint = torch.load(model2load)
    net.load_state_dict(checkpoint['net'], strict=False)

    """ Freeze the model parameters of the pruned_network """
    for param in net.parameters():
        param.requires_grad = False

    return net

def data_loader(batch_size, data):

    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if data == 'mnist':

        train_dataset = datasets.MNIST(root='data', train=True, transform=transform_train, download=True)
        test_dataset = datasets.MNIST(root='data', train=False, transform=transform_test, download=True)

    elif data == 'fmnist':

        train_dataset = datasets.FashionMNIST(root='data', train=True, transform=transform_train, download=True)
        test_dataset = datasets.FashionMNIST(root='data', train=False, transform=transform_test, download=True)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=0,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=0,
                             shuffle=False)

    # Checking the dataset
    for images, labels in train_loader:
        print('Image batch dimensions:', images.shape)
        print('Image label dimensions:', labels.shape)
        break

    return train_loader, test_loader


def compute_accuracy(model, data_loader, device):
    model.eval()
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
        features = features.to(device)
        targets = targets.to(device)

        logits = model(features)
        _, predicted_labels = torch.max(logits, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100


def compute_epoch_loss(model, data_loader, device):
    model.eval()
    curr_loss, num_examples = 0., 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            loss = f.cross_entropy(logits, targets, reduction='sum')
            num_examples += targets.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss

def unnormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def main():

    training = True
    visualize = False
    data = 'mnist' # or 'fmnist'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    """ 1. Load the prune_network and the architecture """
    model2load = 'ckpt_vgg16_prunedto_39,39,63,455,98,97,52,62,22,42,47,47,42,62_90.69.t7'
    # model2load = 'ckpt_vgg16_94.34.t7' # this is pre-trained VGG with CIFAR10 data
    vgg_features = load_pruned_model(model2load, device)

    """ 2. Define a classifier """
    classifier = Classifier(vgg_features).to(device)

    """ 3. Load data to test """
    batch_size = 300
    train_loader, test_loader = data_loader(batch_size, data)

    """ 4. Train the classifier and check the test accuracy """
    if training:
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
        num_epochs = 200

        for epoch in range(num_epochs):

            classifier.train()
            for batch_idx, (features, targets) in enumerate(train_loader):

                features = features.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()

                ### FORWARD AND BACK PROP
                logits = classifier(features)
                loss = criterion(logits, targets)
                loss.backward()

                ### UPDATE MODEL PARAMETERS
                optimizer.step()

                ### LOGGING
                if not batch_idx % 100:
                    print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                          % (epoch + 1, num_epochs, batch_idx,
                             len(train_loader), loss))

            if epoch % 10 == 0:
                classifier.eval()
                with torch.set_grad_enabled(False):  # save memory during inference
                    print('Epoch: %03d/%03d | Train: %.3f%% | Loss: %.3f' % (
                    epoch + 1, num_epochs,
                    compute_accuracy(classifier, train_loader, device),
                    compute_epoch_loss(classifier, train_loader, device)))
                    print('Test accuracy: %.2f%%' % (compute_accuracy(classifier, test_loader, device)))

    """ 5. Plot images with unnormalizing """
    if visualize:
        for batch_idx, (features, targets) in enumerate(test_loader):
            features = features
            break

        n_images = 10
        fig, axes = plt.subplots(nrows=1, ncols=n_images,
                                 sharex=True, sharey=True, figsize=(20, 2.5))
        orig_images = features[:n_images]

        for i in range(n_images):
            curr_img = orig_images[i].detach().to(torch.device('cpu'))
            curr_img = unnormalize(curr_img,
                                   torch.tensor([0.485, 0.456, 0.406]),
                                   torch.tensor([0.229, 0.224, 0.225]))
            curr_img = curr_img.permute((1, 2, 0))
            axes[i].imshow(curr_img)
            # axes[i].set_title(classes[predicted_labels[i]])

        plt.show()

if __name__ == '__main__':
    main()