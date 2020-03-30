import argparse
import torch as pt
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from backpack import backpack, extend
from backpack.extensions import BatchGrad, BatchL2Grad


class LeNet(nn.Module):
  def __init__(self, nodesNum1, nodesNum2, nodesFc1, nodesFc2):
    super(LeNet, self).__init__()

    self.nodesNum2 = nodesNum2

    self.c1 = nn.Conv2d(1, nodesNum1, 5)
    self.s2 = nn.MaxPool2d(2)
    self.c3 = nn.Conv2d(nodesNum1, nodesNum2, 5)
    self.s4 = nn.MaxPool2d(2)
    self.c5 = nn.Linear(nodesNum2 * 4 * 4, nodesFc1)
    self.f6 = nn.Linear(nodesFc1, nodesFc2)
    self.out7 = nn.Linear(nodesFc2, 10)

    # self.parameter = nn.Parameter(-1e-10 * pt.ones(nodesNum1), requires_grad=True)  # this parameter lies #S

  def forward(self, x):
    x = self.c1(x)
    x = nnf.relu(self.s2(x))
    # output = self.bn1(output)
    x = self.c3(x)
    x = nnf.relu(self.s4(x))
    # output = self.bn2(output)
    x = x.view(-1, self.nodesNum2 * 4 * 4)
    x = self.c5(x)
    x = nnf.relu(x)
    x = self.f6(x)
    x = nnf.relu(x)
    x = self.out7(x)  # remove for 99.27 and 90.04 models
    x = nnf.log_softmax(x, dim=1)
    return x


def train(model, device, train_loader, optimizer, epoch, clip_norm, dp_sigma, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = loss_fun(output, target)
        loss = nnf.nll_loss(output, target)

        with backpack(BatchGrad(), BatchL2Grad()):
          loss.backward()

        squared_param_norms = [p.batch_l2 for p in model.parameters()]
        global_norms = pt.sqrt(pt.sum(pt.stack(squared_param_norms), dim=0))
        global_clips = pt.clamp_max(clip_norm / global_norms, 1.)

        for p in model.parameters():
          # clip samplewise gradients, then take average
          clipped_grad = pt.mean(p.grad_batch * global_clips[(...,) + (None,) * (len(p.grad_batch.shape) - 1)], dim=0)
          # compute Gaussian noise standard deviation based on sigma, norm clip & batch size
          noise_sdev = (2 * dp_sigma * clip_norm / clipped_grad.shape[0])
          # clipped_grad = p.grad
          p.grad = clipped_grad + pt.rand_like(clipped_grad, device=device) * noise_sdev

        optimizer.step()
        if batch_idx % log_interval == 0:
            avg_norm, max_norm = pt.mean(global_norms), pt.max(global_norms)
            avg_clipped = pt.mean(global_clips)
            d_done, d_full = batch_idx * len(data), len(train_loader.dataset)

            print(f'Train Epoch: {epoch} [{d_done}/{d_full}]\tLoss: {loss.item():.6f}'
                  f'\t Norm: Max {max_norm:.6f}, Avg {avg_norm:.6f}, Avg Clips {avg_clipped:.6f}')



def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with pt.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += loss_fun(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += nnf.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', '-bs', type=int, default=500)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=14)
    parser.add_argument('--lr', '-lr',type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=20)

    parser.add_argument('--save-model', action='store_true', default=False)

    parser.add_argument('--lenet-size', '-net', type=str, default='20,50,500,500')
    # suggested pruned sizes: 6,8,40,20  -  5,8,45,15  -  6,7,35,17
    parser.add_argument('--clip-norm', '-clip', type=float, default=None)
    parser.add_argument('--dp-sigma', '-sig', type=float, default=None)


    ar = parser.parse_args()
    use_cuda = not ar.no_cuda and pt.cuda.is_available()

    pt.manual_seed(ar.seed)

    device = pt.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = pt.utils.data.DataLoader(datasets.MNIST('data', train=True, download=False, transform=transform),
                                            batch_size=ar.batch_size, shuffle=True, **kwargs)
    test_loader = pt.utils.data.DataLoader(datasets.MNIST('data', train=False, transform=transform),
                                           batch_size=ar.test_batch_size, shuffle=True, **kwargs)

    model = LeNet(*[int(k) for k in ar.lenet_size.split(',')]).to(device)
    model = extend(model)
    optimizer = optim.Adam(model.parameters(), lr=ar.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=ar.gamma)
    for epoch in range(1, ar.epochs + 1):
        train(model, device, train_loader, optimizer, epoch, ar.clip_norm, ar.dp_sigma, ar.log_interval)
        test(model, device, test_loader)
        scheduler.step()

    if ar.save_model:
        pt.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()