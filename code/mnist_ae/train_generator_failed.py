import torch as pt
from torch.optim.lr_scheduler import StepLR
import argparse
import numpy as np
from mnist_ae.models_ae import ConvEncoder, FCEncoder, FCGenerator
from mnist_ae.aux import rff_gauss, get_mnist_dataloaders, plot_mnist_batch, meddistance


def train(args, enc, gen, device, train_loader, optimizer, epoch, rff_mmd_loss):

  gen.train()
  for batch_idx, (data, _) in enumerate(train_loader):
    # print(pt.max(data), pt.min(data))
    data = data.to(device)

    optimizer.zero_grad()
    data_enc = enc(data)
    gen_enc = enc(gen(gen.get_code(data.shape[0]).to(device)))
    # loss = nnf.binary_cross_entropy(output, data)
    loss = rff_mmd_loss(data_enc, gen_enc)
    loss.backward()
    optimizer.step()
    if batch_idx % args.log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))


def test(args, enc, gen, device, test_loader, rff_mmd_loss, epoch):
  gen.eval()

  test_loss = 0
  with pt.no_grad():
    for data, _ in test_loader:
      data = data.to(device)
      data_enc = enc(data)
      gen_samples = gen(gen.get_code(data.shape[0]).to(device))
      gen_enc = enc(gen_samples)
      test_loss += rff_mmd_loss(data_enc, gen_enc).item()  # sum up batch loss
  test_loss /= len(test_loader.dataset)

  data_enc_batch = data_enc.cpu().numpy()
  med_dist = meddistance(data_enc_batch)
  print(f'med distance for encodings is {med_dist}, heuristic suggests sigma={med_dist**2}')

  plot_mnist_batch(data[:100, ...].cpu().numpy(), 10, 10, f'samples/data_samples_ep{epoch}.png')
  plot_samples = gen_samples[:100, ...].cpu().numpy()
  bs = plot_samples.shape[0]
  plot_mnist_batch(plot_samples, 10, 10, f'samples/gen_samples_ep{epoch}.png')
  lit_samples = plot_samples - np.min(np.reshape(plot_samples, (bs, -1)), axis=1)[:, None, None, None]
  lit_samples = lit_samples / np.max(np.reshape(lit_samples, (bs, -1)), axis=1)[:, None, None, None]
  print(np.max(lit_samples), np.min(lit_samples))
  plot_mnist_batch(lit_samples, 10, 10, f'samples/gen_samples_bright_ep{epoch}.png')

  print('Test set: Average loss: {:.4f}'.format(test_loss))


def main():
  # Training settings
  parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
  parser.add_argument('--batch-size', type=int, default=128)
  parser.add_argument('--test-batch-size', type=int, default=1000)
  parser.add_argument('--epochs', type=int, default=10)
  parser.add_argument('--lr', type=float, default=0.0001)
  parser.add_argument('--gamma', type=float, default=0.7)
  parser.add_argument('--no-cuda', action='store_true', default=False)
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--log-interval', type=int, default=100)
  parser.add_argument('--save-model', action='store_true', default=False)

  parser.add_argument('--conv-ae', action='store_true', default=False)
  parser.add_argument('--d-enc', type=int, default=2)
  parser.add_argument('--d-code', type=int, default=2)
  parser.add_argument('--d-rff', type=int, default=100)
  parser.add_argument('--rff-sigma', type=int, default=96.0)

  args = parser.parse_args()

  args.save_model = True
  args.conv_ae = True

  pt.manual_seed(args.seed)

  use_cuda = not args.no_cuda and pt.cuda.is_available()
  train_loader, test_loader = get_mnist_dataloaders(args, use_cuda)

  device = pt.device("cuda" if use_cuda else "cpu")

  if args.conv_ae:
    enc = ConvEncoder(args.d_enc, nc=(1, 8, 16, 32), extra_conv=True)
    enc.load_state_dict(pt.load(f'mnist_encoder_conv_d_enc_{args.d_enc}.pt'))
  else:
    enc = FCEncoder(args.d_enc)
    enc.load_state_dict(pt.load(f'mnist_encoder_fc_d_enc_{args.d_enc}.pt'))

  enc.to(device)
  gen = FCGenerator(d_code=args.d_code, d_hid=300, extra_layer=True).to(device)

  assert args.d_rff % 2 == 0
  w_freq = pt.tensor(np.random.randn(args.d_rff // 2, args.d_enc) / np.sqrt(args.rff_sigma)).to(pt.float32).to(device)

  def mean_embedding(x):
    return pt.mean(rff_gauss(x, w_freq), dim=0)

  def rff_mmd_loss(data_enc, gen_enc):
    return pt.sum((mean_embedding(data_enc) - mean_embedding(gen_enc)) ** 2)

  optimizer = pt.optim.Adam(list(gen.parameters()), lr=args.lr)
  scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
  for epoch in range(1, args.epochs + 1):
    train(args, enc, gen, device, train_loader, optimizer, epoch, rff_mmd_loss)
    test(args, enc, gen, device, test_loader, rff_mmd_loss, epoch)
    scheduler.step()

  if args.save_model:
    pt.save(gen.state_dict(), f'mnist_generator_fc_d_code_{args.d_code}.pt')


if __name__ == '__main__':
  main()
