import torch as pt
import torch.nn.functional as nnf
from torch.optim.lr_scheduler import StepLR
import argparse
from collections import namedtuple
from backpack import extend, backpack
from backpack.extensions import BatchGrad, BatchL2Grad
from mnist_ae.models import ConvEncoder, ConvDecoder, FCEncoder, FCDecoder
from mnist_ae.aux import get_mnist_dataloaders, plot_mnist_batch


def train(args, enc, dec, device, train_loader, optimizer, epoch, enc_loss, dec_loss, dp_spec):
  enc.train()
  dec.train()
  for batch_idx, (data, _) in enumerate(train_loader):
    # print(pt.max(data), pt.min(data))
    data = data.to(device)
    optimizer.zero_grad()
    data_enc = enc(data)

    reconstruction = dec(data_enc)
    # loss = nnf.binary_cross_entropy(output, data)
    l_enc = enc_loss(data, reconstruction)

    if dp_spec.clip is None:
      l_enc.backward()
    else:
      l_enc.backward(retain_graph=True)  # get grads for encoder

      # wipe grads from decoder:
      for param in dec.parameters():
        param.grad = None

      l_dec = dec_loss(data, dec(data_enc))
      with backpack(BatchGrad(), BatchL2Grad()):
        l_dec.backward()

      # compute global gradient norm from parameter gradient norms
      param_norms = pt.sqrt(pt.sum(pt.stack([p.batch_l2**2 for p in dec.parameters()]), dim=0))
      if batch_idx % 100 == 0:
        print(pt.max(param_norms), pt.mean(param_norms))

      # aggregate samplewise grads, replace normal grad

      for param in dec.parameters():
        clips = pt.clamp_max(dp_spec.clip / param_norms, 1.)  # norm clipping not correct
        # print(pt.max(param.batch_l2), pt.mean(param.batch_l2))
        # print(clips.shape, pt.min(clips), pt.max(param.batch_l2))
        # print(clips.shape)
        clips = clips[:, None] if len(param.grad.shape) == 1 else clips[:, None, None]  # hack for 1d and 2d tensors
        # print(clips.shape)
        # print(param.grad_batch.shape)
        clipped_sample_grads = param.grad_batch * clips
        clipped_grad = pt.mean(clipped_sample_grads, dim=0)

        if dp_spec.noise is not None:
          bs = clipped_grad.shape[0]
          clipped_grad = clipped_grad + pt.rand_like(clipped_grad, device=device) * (dp_spec.noise * dp_spec.clip / bs)
        param.grad = clipped_grad
        # print(param.grad.shape)
        # print(param.grad_batch.shape)
        # print(param.batch_l2.shape)
    optimizer.step()
    if batch_idx % args.log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), l_enc.item()))


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
  parser.add_argument('--batch-size', type=int, default=512)
  parser.add_argument('--test-batch-size', type=int, default=1000)
  parser.add_argument('--epochs', type=int, default=20)
  parser.add_argument('--lr', type=float, default=0.001)
  parser.add_argument('--gamma', type=float, default=0.9)
  parser.add_argument('--no-cuda', action='store_true', default=False)
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--log-interval', type=int, default=100)
  parser.add_argument('--save-model', action='store_true', default=False)

  parser.add_argument('--conv-ae', action='store_true', default=False)
  parser.add_argument('--d-enc', type=int, default=5)
  parser.add_argument('--clip-norm', type=float, default=1e-5)
  parser.add_argument('--noise-factor', type=float, default=10.0)
  parser.add_argument('--clip-per-layer', action='store_true', default=False)  # not used yet
  parser.add_argument('--ce-loss', action='store_true', default=False)

  ar = parser.parse_args()

  assert not ar.clip_per_layer  # not implemented yet
  ar.save_model = True

  pt.manual_seed(ar.seed)

  dp_spec = namedtuple('dp_spec', ['clip', 'noise', 'per_layer'])(ar.clip_norm, ar.noise_factor, ar.clip_per_layer)

  use_cuda = not ar.no_cuda and pt.cuda.is_available()
  train_loader, test_loader = get_mnist_dataloaders(ar.batch_size, ar.test_batch_size,
                                                    use_cuda, normalize=not ar.ce_loss)

  device = pt.device("cuda" if use_cuda else "cpu")

  if ar.conv_ae:
    enc = ConvEncoder(ar.d_enc, nc=(1, 8, 16, 32), extra_conv=True).to(device)
    dec = ConvDecoder(ar.d_enc, nc=(32, 16, 8, 1), extra_conv=True).to(device)
  else:
    enc = FCEncoder(ar.d_enc, d_hid=(300, 100), extra_layer=True).to(device)
    dec = extend(FCDecoder(ar.d_enc, d_hid=(100, 300), extra_layer=True).to(device))

  enc_loss = pt.nn.MSELoss()
  dec_loss = extend(pt.nn.MSELoss())

  optimizer = pt.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=ar.lr)

  scheduler = StepLR(optimizer, step_size=1, gamma=ar.gamma)
  for epoch in range(1, ar.epochs + 1):
    train(ar, enc, dec, device, train_loader, optimizer, epoch, enc_loss, dec_loss, dp_spec)
    test(ar, enc, dec, device, test_loader, epoch)
    scheduler.step()

  if ar.save_model:
    arch = 'conv' if ar.conv_ae else 'fc'
    pt.save(dec.state_dict(), f'models/mnist_dec_{arch}_d_enc_{ar.d_enc}_clip_{dp_spec.clip}_noise_{dp_spec.noise}.pt')
    pt.save(enc.state_dict(), f'models/mnist_enc_{arch}_d_enc_{ar.d_enc}_clip_{dp_spec.clip}_noise_{dp_spec.noise}.pt')


if __name__ == '__main__':
  main()

# d_enc = 2
# clip 1e-5 noise 1.0 lr 0.001 bs 256 Train Epoch: 5 loss: 351
# clip 1e-6 noise 1.0 lr 0.001 bs 256 Train Epoch: 5 loss: 369
# clip 1e-5 noise 1.0 lr 0.003 bs 512 Train Epoch: 5 loss: 350
# clip 1e-5 noise 1.0 lr 0.001 bs 512 Train Epoch: 5 loss: 371

# d_enc = 5
# clip 1e-5 noise 1.0 lr 0.001 bs 512 Train Epoch:  5 loss: 260
# clip 1e-5 noise 1.0 lr 0.001 bs 512 Train Epoch: 20 loss: 251
# clip 1e-5 noise 10. lr 0.001 bs 512 Train Epoch: 20 loss: 460