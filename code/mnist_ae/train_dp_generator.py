import torch as pt
from torch.optim.lr_scheduler import StepLR
import argparse
import numpy as np
from mnist_ae.models import ConvEncoder, ConvDecoder, FCEncoder, FCDecoder, FCLatentGenerator
from mnist_ae.aux import rff_gauss, get_mnist_dataloaders, plot_mnist_batch, meddistance


def train(enc, gen, device, train_loader, optimizer, epoch, rff_mmd_loss, log_interval):

  gen.train()
  for batch_idx, (data, _) in enumerate(train_loader):
    # print(pt.max(data), pt.min(data))
    data = data.to(device)

    optimizer.zero_grad()
    data_enc = enc(data)
    gen_enc = gen(gen.get_code(data.shape[0]).to(device))
    # loss = nnf.binary_cross_entropy(output, data)
    loss = rff_mmd_loss(data_enc, gen_enc)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))


def test(enc, dec, gen, device, test_loader, rff_mmd_loss, epoch, batch_size):
  gen.eval()

  test_loss = 0
  with pt.no_grad():
    for data, _ in test_loader:
      data = data.to(device)
      data_enc = enc(data)
      gen_enc = gen(gen.get_code(data.shape[0]).to(device))
      gen_samples = dec(gen_enc)
      test_loss += rff_mmd_loss(data_enc, gen_enc).item()  # sum up batch loss
  test_loss /= (len(test_loader.dataset) / batch_size)

  data_enc_batch = data_enc.cpu().numpy()
  med_dist = meddistance(data_enc_batch)
  print(f'med distance for encodings is {med_dist}, heuristic suggests sigma={med_dist ** 2}')

  plot_mnist_batch(data[:100, ...].cpu().numpy(), 10, 10, f'samples/data_samples_ep{epoch}.png')
  plot_samples = gen_samples[:100, ...].cpu().numpy()
  plot_mnist_batch(plot_samples, 10, 10, f'samples/gen_samples_ep{epoch}.png')
  # bs = plot_samples.shape[0]
  # lit_samples = plot_samples - np.min(np.reshape(plot_samples, (bs, -1)), axis=1)[:, None, None, None]
  # lit_samples = lit_samples / np.max(np.reshape(lit_samples, (bs, -1)), axis=1)[:, None, None, None]
  # print(np.max(lit_samples), np.min(lit_samples))
  # plot_mnist_batch(lit_samples, 10, 10, f'samples/gen_samples_bright_ep{epoch}.png')
  print('Test set: Average loss: {:.4f}'.format(test_loss))


def main():
  # Training settings
  parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
  parser.add_argument('--batch-size', type=int, default=512)
  parser.add_argument('--test-batch-size', type=int, default=1000)
  parser.add_argument('--epochs', type=int, default=10)
  parser.add_argument('--lr', type=float, default=0.001)
  parser.add_argument('--gamma', type=float, default=0.7)
  parser.add_argument('--no-cuda', action='store_true', default=False)
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--log-interval', type=int, default=100)
  parser.add_argument('--save-model', action='store_true', default=False)

  parser.add_argument('--d-enc', type=int, default=2)
  parser.add_argument('--d-code', type=int, default=2)
  parser.add_argument('--d-rff', type=int, default=100)
  parser.add_argument('--rff-sigma', type=int, default=90.0)

  parser.add_argument('--noise-factor', type=float, default=1.0)

  parser.add_argument('--conv-ae', action='store_true', default=False)
  parser.add_argument('--ae_clip', type=float, default=1e-5)
  parser.add_argument('--ae_noise', type=float, default=1.0)

  ar = parser.parse_args()

  ar.save_model = True
  # ar.conv_ae = True

  pt.manual_seed(ar.seed)

  use_cuda = not ar.no_cuda and pt.cuda.is_available()
  train_loader, test_loader = get_mnist_dataloaders(ar.batch_size, ar.test_batch_size, use_cuda)

  device = pt.device("cuda" if use_cuda else "cpu")

  if ar.conv_ae:
    enc = ConvEncoder(ar.d_enc, nc=(1, 8, 16, 32), extra_conv=True)
    dec = ConvDecoder(ar.d_enc, nc=(32, 16, 8, 1), extra_conv=True)
    enc.load_state_dict(pt.load(f'mnist_encoder_conv_d_enc_{ar.d_enc}.pt'))
    dec.load_state_dict(pt.load(f'mnist_decoder_conv_d_enc_{ar.d_enc}.pt'))
    raise NotImplementedError
  else:
    enc = FCEncoder(ar.d_enc, d_hid=(300, 100), extra_layer=True)
    dec = FCDecoder(ar.d_enc, d_hid=(100, 300), extra_layer=True)
    enc.load_state_dict(pt.load(f'models/mnist_enc_fc_d_enc_{ar.d_enc}_clip_{ar.ae_clip}_noise_{ar.ae_noise}.pt'))
    dec_extended_dict = pt.load(f'models/mnist_dec_fc_d_enc_{ar.d_enc}_clip_{ar.ae_clip}_noise_{ar.ae_noise}.pt')
    dec_reduced_dict = {k: dec_extended_dict[k] for k in dec.state_dict().keys()}
    dec.load_state_dict(dec_reduced_dict)

  enc.to(device)
  dec.to(device)
  gen = FCLatentGenerator(d_code=ar.d_code, d_enc=ar.d_enc, d_hid=100, extra_layer=True).to(device)

  assert ar.d_rff % 2 == 0
  w_freq = pt.tensor(np.random.randn(ar.d_rff // 2, ar.d_enc) / np.sqrt(ar.rff_sigma)).to(pt.float32).to(device)

  def mean_embedding(x):
    return pt.mean(rff_gauss(x, w_freq), dim=0)

  def rff_mmd_loss(data_enc, gen_enc):
    data_emb = mean_embedding(data_enc) + pt.randn(ar.d_rff, device=device) * (ar.noise_factor / ar.batch_size)
    gen_emb = mean_embedding(gen_enc)
    return pt.sum((data_emb - gen_emb) ** 2)

  optimizer = pt.optim.Adam(list(gen.parameters()), lr=ar.lr)
  scheduler = StepLR(optimizer, step_size=1, gamma=ar.gamma)
  for epoch in range(1, ar.epochs + 1):
    train(enc, gen, device, train_loader, optimizer, epoch, rff_mmd_loss, ar.log_interval)
    test(enc, dec, gen, device, test_loader, rff_mmd_loss, epoch, ar.batch_size)
    scheduler.step()

  if ar.save_model:
    pt.save(gen.state_dict(), f'models/mnist_gen_fc_d_code_{ar.d_code}_d_enc_{ar.d_enc}_noise_{ar.noise_factor}.pt')


if __name__ == '__main__':
  main()
