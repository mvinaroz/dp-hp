import torch as pt
import torch.nn.functional as nnf
from torch.optim.lr_scheduler import StepLR
import argparse
from mnist_ae.models import ConvEncoder, ConvDecoder, FCEncoder, FCDecoder
from mnist_ae.aux import get_mnist_dataloaders, plot_mnist_batch


def train(args, enc, dec, device, train_loader, optimizer, epoch):
  enc.train()
  dec.train()
  for batch_idx, (data, _) in enumerate(train_loader):
    # print(pt.max(data), pt.min(data))
    data = data.to(device)
    optimizer.zero_grad()
    output = dec(enc(data))
    # loss = nnf.binary_cross_entropy(output, data)
    loss = nnf.mse_loss(output, data)
    loss.backward()
    optimizer.step()
    if batch_idx % args.log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))


def test(args, enc, dec, device, test_loader, epoch):
  enc.eval()
  dec.eval()
  test_loss = 0
  with pt.no_grad():
    for data, _ in test_loader:
      data = data.to(device)
      output = dec(enc(data))
      test_loss += nnf.mse_loss(output, data, reduction='sum').item()  # sum up batch loss
  test_loss /= len(test_loader.dataset)

  plot_mnist_batch(data[:100, ...].cpu().numpy(), 10, 10, f'samples/ae_data_samples_ep{epoch}.png')
  plot_mnist_batch(output[:100, ...].cpu().numpy(), 10, 10, f'samples/ae_reconstructions_ep{epoch}.png')


  print('Test set: Average loss: {:.4f}'.format(test_loss))


def main():
  # Training settings
  parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--test-batch-size', type=int, default=1000)
  parser.add_argument('--epochs', type=int, default=5)
  parser.add_argument('--lr', type=float, default=0.001)
  parser.add_argument('--gamma', type=float, default=0.7)
  parser.add_argument('--no-cuda', action='store_true', default=False)
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--log-interval', type=int, default=100)
  parser.add_argument('--save-model', action='store_true', default=False)

  parser.add_argument('--conv-ae', action='store_true', default=False)
  parser.add_argument('--d-enc', type=int, default=2)
  args = parser.parse_args()

  args.save_model = True
  args.conv_ae = True

  pt.manual_seed(args.seed)

  use_cuda = not args.no_cuda and pt.cuda.is_available()
  train_loader, test_loader = get_mnist_dataloaders(args, use_cuda)

  device = pt.device("cuda" if use_cuda else "cpu")

  if args.conv_ae:
    enc = ConvEncoder(args.d_enc, nc=(1, 8, 16, 32), extra_conv=True).to(device)
    dec = ConvDecoder(args.d_enc, nc=(32, 16, 8, 1), extra_conv=True).to(device)
  else:
    enc = FCEncoder(args.d_enc).to(device)
    dec = FCDecoder(args.d_enc).to(device)

  optimizer = pt.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=args.lr)

  scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
  for epoch in range(1, args.epochs + 1):
    train(args, enc, dec, device, train_loader, optimizer, epoch)
    test(args, enc, dec, device, test_loader, epoch)
    scheduler.step()

  if args.save_model:
    pt.save(dec.state_dict(), "mnist_decoder_{}_d_enc_{}.pt".format('conv' if args.conv_ae else 'fc', args.d_enc))
    pt.save(enc.state_dict(), "mnist_encoder_{}_d_enc_{}.pt".format('conv' if args.conv_ae else 'fc', args.d_enc))


if __name__ == '__main__':
  main()

# Save:
# torch.save(model.state_dict(), PATH)

# Load:
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()
