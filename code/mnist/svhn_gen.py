import os
import torch as pt
from torch.optim.lr_scheduler import StepLR
import argparse
import numpy as np
from models_gen import ConvCondGenSVHN, ConvGenSVHN
from aux import rff_gauss, get_svhn_dataloaders, plot_svhn_batch, meddistance, save_gen_labels, log_args, flat_data
from real_mmd_loss import mmd_g, get_squared_dist


def get_single_release_loss(train_loader, d_enc, d_rff, rff_sigma, device, do_gen_labels, n_labels, noise_factor):
  # create data mean embedding and rff_loss function
  assert d_rff % 2 == 0
  w_freq = pt.tensor(np.random.randn(d_rff // 2, d_enc) / np.sqrt(rff_sigma)).to(pt.float32).to(device)

  emb_acc = []
  n_data = 0
  for data, labels in train_loader:

    # print(pt.max(data), pt.min(data))
    data, labels = data.to(device), labels.to(device)
    data = flat_data(data, labels, device, n_labels=10, add_label=False)
    data_emb = rff_gauss(data, w_freq)
    if not do_gen_labels:
      emb_acc.append(pt.sum(data_emb, 0))
    else:
      one_hots = pt.zeros(labels.shape[0], 10, device=device)
      one_hots.scatter_(1, labels[:, None], 1)
      emb_acc.append(pt.sum(pt.einsum('ki,kj->kij', [rff_gauss(data, w_freq), one_hots]), 0))
    n_data += data.shape[0]

  print('done collecting batches, n_data', n_data)
  emb_acc = pt.sum(pt.stack(emb_acc), 0) / n_data
  print(pt.norm(emb_acc), emb_acc.shape)
  if not do_gen_labels:
    noise = pt.randn(d_rff, device=device) * (2 * noise_factor / n_data)
  else:
    noise = pt.randn(d_rff, n_labels, device=device) * (2 * noise_factor / n_data)
  noisy_emb = emb_acc + noise

  if not do_gen_labels:

    def mean_embedding(x):
      return pt.mean(rff_gauss(x, w_freq), dim=0)

    def rff_mmd_loss(gen_out):
      gen_emb = mean_embedding(gen_out)
      return pt.sum((noisy_emb - gen_emb) ** 2)
  else:
    def label_mean_embedding(data, labels):
      return pt.mean(pt.einsum('ki,kj->kij', [rff_gauss(data, w_freq), labels]), 0)

    def rff_mmd_loss(gen_enc, gen_labels):
      gen_emb = label_mean_embedding(gen_enc, gen_labels)
      return pt.sum((noisy_emb - gen_emb) ** 2)

  return rff_mmd_loss, noisy_emb


def train_single_release(gen, device, optimizer, epoch, rff_mmd_loss, log_interval, do_gen_labels, batch_size, n_data):
  n_iter = n_data // batch_size
  for batch_idx in range(n_iter):
    # print(pt.max(data), pt.min(data))
    if not do_gen_labels:
      loss = rff_mmd_loss(gen(gen.get_code(batch_size, device)))

    else:
      # one_hots = pt.zeros(bs, 10, device=device)
      # one_hots.scatter_(1, labels.to(device)[:, None], 1)
      gen_code, gen_labels = gen.get_code(batch_size, device)
      loss = rff_mmd_loss(gen(gen_code), gen_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
        epoch, batch_idx * batch_size, n_data, loss.item()))


def train_multi_release(gen, device, train_loader, optimizer, epoch, rff_mmd_loss, log_interval, do_gen_labels):

  for batch_idx, (data, labels) in enumerate(train_loader):
    # print(pt.max(data), pt.min(data))
    data, labels = data.to(device), labels.to(device)
    data = flat_data(data, labels, device, n_labels=10, add_label=False)

    bs = labels.shape[0]
    if not do_gen_labels:
      loss = rff_mmd_loss(data, gen(gen.get_code(bs, device)))

    else:
      one_hots = pt.zeros(bs, 10, device=device)
      one_hots.scatter_(1, labels[:, None], 1)
      gen_code, gen_labels = gen.get_code(bs, device)
      loss = rff_mmd_loss(data, one_hots, gen(gen_code), gen_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()))


def test(gen, device, test_loader, rff_mmd_loss, epoch, batch_size, do_gen_labels, log_dir):
  test_loss = 0
  gen_labels, ordered_labels = None, None
  plot_data = None
  with pt.no_grad():
    for data, labels in test_loader:
      if plot_data is None:
        plot_data = data[:100, ...].numpy()

      data = data.to(device)
      data = flat_data(data.to(device), labels.to(device), device, n_labels=10, add_label=False)

      bs = labels.shape[0]

      if not do_gen_labels:
        gen_samples = gen(gen.get_code(bs, device))
        gen_labels = None
        loss = rff_mmd_loss(data, gen_samples)

      else:
        one_hots = pt.zeros(bs, 10, device=device)
        one_hots.scatter_(1, labels.to(device)[:, None], 1)
        gen_code, gen_labels = gen.get_code(bs, device)
        gen_samples = gen(gen_code)
        loss = rff_mmd_loss(data, one_hots, gen_samples, gen_labels)

      test_loss += loss.item()  # sum up batch loss
  test_loss /= (len(test_loader.dataset) / batch_size)

  data_enc_batch = data.cpu().numpy()
  med_dist = meddistance(data_enc_batch)
  print(f'med distance for encodings is {med_dist}, heuristic suggests sigma={med_dist ** 2}')

  print(plot_data.shape)
  plot_svhn_batch(plot_data, 10, 10, log_dir + f'data_ep{epoch}')

  if do_gen_labels:
    ordered_labels = pt.repeat_interleave(pt.arange(10), 10)[:, None].to(device)
    gen_code, gen_labels = gen.get_code(100, device, labels=ordered_labels)
    gen_samples = gen(gen_code).detach()

  plot_samples = gen_samples[:100, ...].cpu().numpy()
  plot_svhn_batch(plot_samples, 10, 10, log_dir + f'samples_ep{epoch}')
  if gen_labels is not None and ordered_labels is None:
    save_gen_labels(gen_labels[:100, ...].cpu().numpy(), 10, 10, log_dir + f'labels_ep{epoch}')
  print('Test set: Average loss: {:.4f}'.format(test_loss))


def get_rff_mmd_loss(d_enc, d_rff, rff_sigma, device, do_gen_labels, n_labels, noise_factor, batch_size, real_mmd,
                     renorm_release):
  assert d_rff % 2 == 0
  w_freq = pt.tensor(np.random.randn(d_rff // 2, d_enc) / np.sqrt(rff_sigma)).to(pt.float32).to(device)

  if real_mmd:
    # print('we use full MMD here')

    def rff_mmd_loss(data_enc, gen_enc):
      dxx, dxy, dyy = get_squared_dist(data_enc, gen_enc)
      return mmd_g(dxx, dxy, dyy, batch_size, sigma=np.sqrt(rff_sigma))

  elif not do_gen_labels:

    def mean_embedding(x):
      return pt.mean(rff_gauss(x, w_freq), dim=0)

    def rff_mmd_loss(data_enc, gen_out):
      data_emb = mean_embedding(data_enc)
      gen_emb = mean_embedding(gen_out)
      noise = pt.randn(d_rff, device=device) * (2 * noise_factor / batch_size)
      noisy_emb = data_emb + noise
      if renorm_release == 'scale':
        noisy_emb = noisy_emb / pt.norm(noisy_emb)  # rescale to L2 norm 1
      elif renorm_release == 'clip':
        noisy_emb = noisy_emb / pt.max(pt.norm(noisy_emb), other=pt.tensor(1., device=device))  # clip to L2 norm 1, don't touch smaller norms
      return pt.sum((noisy_emb - gen_emb) ** 2)
  else:
    # print('we use the random feature mean embeddings here')

    def label_mean_embedding(data, labels):
      return pt.mean(pt.einsum('ki,kj->kij', [rff_gauss(data, w_freq), labels]), 0)

    def rff_mmd_loss(data_enc, labels, gen_enc, gen_labels):
      data_emb = label_mean_embedding(data_enc, labels)  # (d_rff, n_labels)
      gen_emb = label_mean_embedding(gen_enc, gen_labels)
      noise = pt.randn(d_rff, n_labels, device=device) * (2 * noise_factor / batch_size)
      noisy_emb = data_emb + noise
      if renorm_release == 'scale':
        noisy_emb = noisy_emb / pt.norm(noisy_emb)  # rescale to L2 norm 1
      elif renorm_release == 'clip':
        noisy_emb = noisy_emb / pt.max(pt.norm(noisy_emb), other=pt.tensor(1., device=device))  # clip to L2 norm 1, don't touch smaller norms
      return pt.sum((noisy_emb - gen_emb) ** 2)
  return rff_mmd_loss


def get_args():
  parser = argparse.ArgumentParser()

  # BASICS
  parser.add_argument('--no-cuda', action='store_true', default=False, help='turns off gpu')
  parser.add_argument('--seed', type=int, default=None, help='sets random seed')
  parser.add_argument('--n-labels', type=int, default=10, help='number of labels/classes in data')
  parser.add_argument('--log-interval', type=int, default=100, help='print updates after n steps')
  parser.add_argument('--base-log-dir', type=str, default='logs/gen/', help='path where logs for all runs are stored')
  parser.add_argument('--log-name', type=str, default=None, help='subdirectory for this run')
  parser.add_argument('--log-dir', type=str, default=None, help='override save path. constructed if None')
  parser.add_argument('--data', type=str, default='digits', help='options are digits and fashion')
  parser.add_argument('--synth-svhn', action='store_true', default=False, help='if true, make 60k synthetic mnist')

  # OPTIMIZATION
  parser.add_argument('--batch-size', '-bs', type=int, default=100)
  parser.add_argument('--test-batch-size', '-tbs', type=int, default=200)
  parser.add_argument('--epochs', '-ep', type=int, default=5)
  parser.add_argument('--lr', '-lr', type=float, default=1e-4, help='learning rate')
  parser.add_argument('--lr-decay', type=float, default=0.9, help='per epoch learning rate decay factor')

  # MODEL DEFINITION
  parser.add_argument('--batch-norm', action='store_true', default=False, help='use batch norm in model')
  parser.add_argument('--gen-labels', action='store_true', default=False, help='generate labels as well as samples')
  parser.add_argument('--d-code', '-dcode', type=int, default=5, help='random code dimensionality')
  parser.add_argument('--gen-spec', type=str, default='200,100', help='specifies hidden layers of generator')
  parser.add_argument('--real-mmd', action='store_true', default=True, help='for debug: dont approximate mmd')
  parser.add_argument('--conv-gen', action='store_true', default=True, help='use convolutional generator')
  parser.add_argument('--kernel-sizes', '-ks', type=str, default='3,5,5', help='specifies conv gen kernel sizes')
  parser.add_argument('--n-channels', '-nc', type=str, default='32,16,8', help='specifies conv gen kernel sizes')

  # DP SPEC
  parser.add_argument('--d-rff', type=int, default=10000, help='number of random filters for apprixmate mmd')
  parser.add_argument('--rff-sigma', '-rffsig', type=float, default=200.0, help='standard dev. for filter sampling')
  parser.add_argument('--noise-factor', '-noise', type=float, default=0.0, help='privacy noise parameter')
  parser.add_argument('--renorm-release', type=str, default=None, help='project noisy loss back to hypersphere')

  parser.add_argument('--single-release', action='store_true', default=False, help='get 1 data mean embedding only')
  ar = parser.parse_args()

  if ar.log_dir is None:
    ar.log_dir = get_log_dir(ar)
  if not os.path.exists(ar.log_dir):
    os.makedirs(ar.log_dir)
  preprocess_args(ar)
  log_args(ar.log_dir, ar)
  return ar


def get_log_dir(ar):
  if ar.log_name is not None:
    log_dir = ar.base_log_dir + ar.log_name + '/'
  else:
    gen_type = f'{"labeled_" if ar.gen_labels else "unlabeled_"}'
    print('gen_type', gen_type)
    gen_spec = f'gen{ar.gen_spec}_sig{ar.noise_factor}_dcode{ar.d_code}_drff{ar.d_rff}_rffsig{ar.rff_sigma}_bs{ar.batch_size}'
    print('gen_spec', gen_spec)
    log_dir = ar.base_log_dir + gen_type + gen_spec + '/'
  return log_dir


def preprocess_args(args):
  if args.seed is None:
    args.seed = np.random.randint(0, 1000)

  assert args.data in {'digits', 'fashion'}
  assert not (args.gen_labels and args.real_mmd)
  assert not (args.real_mmd and args.single_release)


def synthesize_mnist_with_uniform_labels(gen, device, gen_batch_size=1000, n_data=60000, n_labels=10):
  gen.eval()
  assert n_data % gen_batch_size == 0
  assert gen_batch_size % n_labels == 0
  n_iterations = n_data // gen_batch_size

  data_list = []
  ordered_labels = pt.repeat_interleave(pt.arange(n_labels), gen_batch_size // n_labels)[:, None].to(device)
  labels_list = [ordered_labels] * n_iterations

  with pt.no_grad():
    for idx in range(n_iterations):
      gen_code, gen_labels = gen.get_code(gen_batch_size, device, labels=ordered_labels)
      gen_samples = gen(gen_code)
      data_list.append(gen_samples)
  return pt.cat(data_list, dim=0).cpu().numpy(), pt.cat(labels_list, dim=0).cpu().numpy()


def main():
  # Training settings

  ar = get_args()
  pt.manual_seed(ar.seed)

  use_cuda = not ar.no_cuda and pt.cuda.is_available()
  train_loader, test_loader = get_svhn_dataloaders(ar.batch_size, ar.test_batch_size, use_cuda)

  device = pt.device("cuda" if use_cuda else "cpu")

  # gen_spec = [int(k) for k in ar.gen_spec.split(',')] if ar.gen_spec is not None else None
  if ar.gen_labels:
    gen = ConvCondGenSVHN(ar.d_code, ar.gen_spec, ar.n_labels, ar.n_channels, ar.kernel_sizes, use_sigmoid=True)
  else:
    gen = ConvGenSVHN(ar.d_code, ar.gen_spec, ar.n_labels, ar.n_channels, ar.kernel_sizes, use_sigmoid=True)
    
  gen = gen.to(device)

  svhn_dims = 3*32**2

  if ar.single_release:
    single_release_loss, _ = get_single_release_loss(train_loader, svhn_dims, ar.d_rff, ar.rff_sigma, device, ar.gen_labels,
                                                     ar.n_labels, ar.noise_factor)
  else:
    single_release_loss = None
  rff_mmd_loss = get_rff_mmd_loss(svhn_dims, ar.d_rff, ar.rff_sigma, device, ar.gen_labels,
                                  ar.n_labels, ar.noise_factor, ar.batch_size, ar.real_mmd, ar.renorm_release)

  optimizer = pt.optim.Adam(list(gen.parameters()), lr=ar.lr)
  scheduler = StepLR(optimizer, step_size=1, gamma=ar.lr_decay)

  for epoch in range(1, ar.epochs + 1):
    if ar.single_release:
      train_single_release(gen, device, optimizer, epoch, single_release_loss, ar.log_interval, ar.gen_labels,
                           ar.batch_size, n_data=73257)
    else:
      train_multi_release(gen, device, train_loader, optimizer, epoch, rff_mmd_loss, ar.log_interval,
                          ar.gen_labels)
    test(gen, device, test_loader, rff_mmd_loss, epoch, ar.batch_size,
         ar.gen_labels, ar.log_dir)
    scheduler.step()

  pt.save(gen.state_dict(), ar.log_dir + 'gen.pt')
  if ar.synth_svhn:
    assert ar.uniform_labels
    # syn_data, syn_labels = synthesize_mnist_with_uniform_labels(gen, device)
    # np.savez(ar.log_dir + 'synthetic_mnist', data=syn_data, labels=syn_labels)


if __name__ == '__main__':
  main()
