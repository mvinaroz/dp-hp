import os
import torch as pt
from torch.optim.lr_scheduler import StepLR
import argparse
import numpy as np
from models_gen import FCCondGen, ConvCondGen
from aux import get_mnist_dataloaders, plot_mnist_batch, meddistance, log_args, flat_data, log_final_score
from mmd_approx import rff_sphere, weights_sphere
from synth_data_benchmark import test_gen_data
from real_mmd_loss import get_real_mmd_loss
from kmeans import get_kmeans_mmd_loss


def data_label_embedding(data, labels, w_freq, labels_to_one_hot=False, device=None, reduce='mean'):
  assert reduce in {'mean', 'sum'}
  if labels_to_one_hot:
    batch_size = data.shape[0]
    one_hots = pt.zeros(batch_size, 10, device=device)
    one_hots.scatter_(1, labels[:, None], 1)
    labels = one_hots
  embedding = pt.einsum('ki,kj->kij', [rff_sphere(data, w_freq), labels])
  return pt.mean(embedding, 0) if reduce == 'mean' else pt.sum(embedding, 0)


def get_single_release_loss(train_loader, d_enc, d_rff, rff_sigma, device, n_labels, noise_factor):
  assert d_rff % 2 == 0
  # w_freq = pt.tensor(np.random.randn(d_rff // 2, d_enc) / np.sqrt(rff_sigma)).to(pt.float32).to(device)
  w_freq = weights_sphere(d_rff, d_enc, rff_sigma, device)

  emb_acc = []
  n_data = 0
  for data, labels in train_loader:
    data, labels = data.to(device), labels.to(device)
    data = flat_data(data, labels, device, n_labels=10, add_label=False)

    emb_acc.append(data_label_embedding(data, labels, w_freq, labels_to_one_hot=True, device=device, reduce='sum'))
    # emb_acc.append(pt.sum(pt.einsum('ki,kj->kij', [rff_gauss(data, w_freq), one_hots]), 0))
    n_data += data.shape[0]

  print('done collecting batches, n_data', n_data)
  emb_acc = pt.sum(pt.stack(emb_acc), 0) / n_data
  print(pt.norm(emb_acc), emb_acc.shape)
  noise = pt.randn(d_rff, n_labels, device=device) * (2 * noise_factor / n_data)
  noisy_emb = emb_acc + noise

  def rff_mmd_loss(gen_data, gen_labels):
    gen_emb = data_label_embedding(gen_data, gen_labels, w_freq)
    return pt.sum((noisy_emb - gen_emb) ** 2)

  return rff_mmd_loss, noisy_emb


def get_rff_mmd_loss(d_enc, d_rff, rff_sigma, device, n_labels, noise_factor, batch_size):
  assert d_rff % 2 == 0
  # w_freq = pt.tensor(np.random.randn(d_rff // 2, d_enc) / np.sqrt(rff_sigma)).to(pt.float32).to(device)
  w_freq = weights_sphere(d_rff, d_enc, rff_sigma, device)

  def rff_mmd_loss(data_enc, labels, gen_enc, gen_labels):
    data_emb = data_label_embedding(data_enc, labels, w_freq, labels_to_one_hot=True, device=device)  # (d_rff, n_labels)
    gen_emb = data_label_embedding(gen_enc, gen_labels, w_freq)
    noise = pt.randn(d_rff, n_labels, device=device) * (2 * noise_factor / batch_size)
    noisy_emb = data_emb + noise
    return pt.sum((noisy_emb - gen_emb) ** 2)

  return rff_mmd_loss


def train_single_release(gen, device, optimizer, epoch, rff_mmd_loss, log_interval, batch_size, n_data):
  n_iter = n_data // batch_size
  for batch_idx in range(n_iter):
    gen_code, gen_labels = gen.get_code(batch_size, device)
    loss = rff_mmd_loss(gen(gen_code), gen_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * batch_size, n_data, loss.item()))


def compute_rff_loss(gen, data, labels, rff_mmd_loss, device):
  bs = labels.shape[0]
  gen_code, gen_labels = gen.get_code(bs, device)
  gen_samples = gen(gen_code)
  return rff_mmd_loss(data, labels, gen_samples, gen_labels)


def train_multi_release(gen, device, train_loader, optimizer, epoch, rff_mmd_loss, log_interval):

  for batch_idx, (data, labels) in enumerate(train_loader):
    data, labels = data.to(device), labels.to(device)
    data = flat_data(data, labels, device, n_labels=10, add_label=False)

    loss = compute_rff_loss(gen, data, labels, rff_mmd_loss, device)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_idx % log_interval == 0:
      n_data = len(train_loader.dataset)
      print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), n_data, loss.item()))


def test(gen, device, test_loader, rff_mmd_loss, epoch, batch_size, log_dir):
  test_loss = 0
  with pt.no_grad():
    for data, labels in test_loader:
      data, labels = data.to(device), labels.to(device)
      data = flat_data(data, labels, device, n_labels=10, add_label=False)
      loss = compute_rff_loss(gen, data, labels, rff_mmd_loss, device)
      test_loss += loss.item()  # sum up batch loss

  test_loss /= (len(test_loader.dataset) / batch_size)

  data_enc_batch = data.cpu().numpy()
  med_dist = meddistance(data_enc_batch)
  print(f'med distance for encodings is {med_dist}, heuristic suggests sigma={med_dist ** 2}')

  ordered_labels = pt.repeat_interleave(pt.arange(10), 10)[:, None].to(device)
  gen_code, gen_labels = gen.get_code(100, device, labels=ordered_labels)
  gen_samples = gen(gen_code).detach()

  plot_samples = gen_samples[:100, ...].cpu().numpy()
  plot_mnist_batch(plot_samples, 10, 10, log_dir + f'samples_ep{epoch}', denorm=False)
  print('Test set: Average loss: {:.4f}'.format(test_loss))


def get_args():
  parser = argparse.ArgumentParser()

  # BASICS
  parser.add_argument('--seed', type=int, default=None, help='sets random seed')
  parser.add_argument('--n-labels', type=int, default=10, help='number of labels/classes in data')
  parser.add_argument('--log-interval', type=int, default=100, help='print updates after n steps')
  parser.add_argument('--base-log-dir', type=str, default='logs/gen/', help='path where logs for all runs are stored')
  parser.add_argument('--log-name', type=str, default=None, help='subdirectory for this run')
  parser.add_argument('--log-dir', type=str, default=None, help='override save path. constructed if None')
  parser.add_argument('--data', type=str, default='digits', help='options are digits and fashion')
  parser.add_argument('--synth-mnist', action='store_true', default=True, help='if true, make 60k synthetic mnist')

  # OPTIMIZATION
  parser.add_argument('--batch-size', '-bs', type=int, default=500)
  parser.add_argument('--test-batch-size', '-tbs', type=int, default=1000)
  parser.add_argument('--epochs', '-ep', type=int, default=5)
  parser.add_argument('--lr', '-lr', type=float, default=0.01, help='learning rate')
  parser.add_argument('--lr-decay', type=float, default=0.9, help='per epoch learning rate decay factor')

  # MODEL DEFINITION
  # parser.add_argument('--batch-norm', action='store_true', default=True, help='use batch norm in model')
  parser.add_argument('--conv-gen', action='store_true', default=True, help='use convolutional generator')
  parser.add_argument('--d-code', '-dcode', type=int, default=5, help='random code dimensionality')
  parser.add_argument('--gen-spec', type=str, default='200', help='specifies hidden layers of generator')
  parser.add_argument('--kernel-sizes', '-ks', type=str, default='5,5', help='specifies conv gen kernel sizes')
  parser.add_argument('--n-channels', '-nc', type=str, default='16,8', help='specifies conv gen kernel sizes')

  # DP SPEC
  parser.add_argument('--d-rff', type=int, default=1000, help='number of random filters for apprixmate mmd')
  parser.add_argument('--rff-sigma', '-rffsig', type=float, default=None, help='standard dev. for filter sampling')
  parser.add_argument('--noise-factor', '-noise', type=float, default=5.0, help='privacy noise parameter')

  # ALTERNATE MODES
  parser.add_argument('--single-release', action='store_true', default=False, help='get 1 data mean embedding only')
  parser.add_argument('--real-mmd', action='store_true', default=False, help='for debug: dont approximate mmd')

  parser.add_argument('--kmeans-mmd', action='store_true', default=False, help='for debug: dont approximate mmd')
  parser.add_argument('--n-means', type=int, default=10, help='number of means to find per class')
  parser.add_argument('--dp-kmeans-encoding-dim', type=int, default=10, help='dimension the data is projected to')
  parser.add_argument('--tgt-epsilon', type=float, default=1000.0, help='privacy epsilon for dp k-means')
  parser.add_argument('--kmeans-delta', type=float, default=0.01, help='soft failure probability in dp k-means')

  parser.add_argument('--center-data', action='store_true', default=False, help='k-means requires centering')

  ar = parser.parse_args()

  preprocess_args(ar)
  log_args(ar.log_dir, ar)
  return ar


def preprocess_args(ar):
  if ar.log_dir is None:
    assert ar.log_name is not None
    ar.log_dir = ar.base_log_dir + ar.log_name + '/'
  if not os.path.exists(ar.log_dir):
    os.makedirs(ar.log_dir)

  if ar.seed is None:
    ar.seed = np.random.randint(0, 1000)
  assert ar.data in {'digits', 'fashion'}
  if ar.rff_sigma is None:
    ar.rff_sigma = 105 if ar.data == 'digits' else 127

  if ar.kmeans_mmd and ar.tgt_epsilon > 0.0:
    assert ar.center_data, 'dp kmeans requires centering of data'


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
  # load settings
  ar = get_args()
  pt.manual_seed(ar.seed)
  use_cuda = pt.cuda.is_available()
  device = pt.device("cuda" if use_cuda else "cpu")

  # load data
  train_loader, test_loader = get_mnist_dataloaders(ar.batch_size, ar.test_batch_size, use_cuda, dataset=ar.data,
                                                    normalize=ar.center_data)

  # init model
  if ar.conv_gen:
    gen = ConvCondGen(ar.d_code, ar.gen_spec, ar.n_labels, ar.n_channels, ar.kernel_sizes).to(device)
  else:
    gen = FCCondGen(ar.d_code, ar.gen_spec, ar.n_labels, batch_norm=True).to(device)

  # define loss function
  if ar.single_release:
    single_release_loss, _ = get_single_release_loss(train_loader, 784, ar.d_rff, ar.rff_sigma, device, ar.n_labels,
                                                     ar.noise_factor)
  elif ar.kmeans_mmd:
    single_release_loss = get_kmeans_mmd_loss(train_loader, ar.n_labels, ar.noise_factor, ar.n_means,
                                              ar.rff_sigma, ar.batch_size, ar.dp_kmeans_encoding_dim)

  else:
    single_release_loss = None

  if ar.real_mmd:
    rff_mmd_loss = get_real_mmd_loss(ar.rff_sigma, ar.n_labels, ar.batch_size)
  else:
    rff_mmd_loss = get_rff_mmd_loss(784, ar.d_rff, ar.rff_sigma, device, ar.n_labels, ar.noise_factor, ar.batch_size)

  # init optimizer
  optimizer = pt.optim.Adam(list(gen.parameters()), lr=ar.lr)
  scheduler = StepLR(optimizer, step_size=1, gamma=ar.lr_decay)

  # training loop
  for epoch in range(1, ar.epochs + 1):
    if ar.single_release:
      train_single_release(gen, device, optimizer, epoch, single_release_loss, ar.log_interval, ar.batch_size, 60000)
    else:
      train_multi_release(gen, device, train_loader, optimizer, epoch, rff_mmd_loss, ar.log_interval)

    # testing doesn't really inform how training is going, so it's commented out
    # test(gen, device, test_loader, rff_mmd_loss, epoch, ar.batch_size, ar.log_dir)
    scheduler.step()

  # save trained model and data
  pt.save(gen.state_dict(), ar.log_dir + 'gen.pt')
  if ar.synth_mnist:
    syn_data, syn_labels = synthesize_mnist_with_uniform_labels(gen, device)
    np.savez(ar.log_dir + 'synthetic_mnist', data=syn_data, labels=syn_labels)

    final_score = test_gen_data(ar.log_name, ar.data, subsample=0.1, custom_keys='logistic_reg')
    log_final_score(ar.log_dir, final_score)


if __name__ == '__main__':
  main()
