import os
import torch as pt
from torch.optim.lr_scheduler import StepLR
import argparse
import numpy as np
from models_gen import ConvCondGen
from aux import plot_mnist_batch, meddistance, log_args, flat_data, log_final_score
from data_loading import get_mnist_dataloaders
from rff_mmd_approx import get_rff_losses, apply_pca
from synth_data_benchmark import test_gen_data


import numpy as np


def private_pca(x_data, sigma, pca_dims):
    x_data = np.float32(x_data)
    n_samples = x_data.shape[0]
    x_dims = x_data.shape[1]

    x_data_norms = np.linalg.norm(x_data, 2, axis=1, keepdims=True)
    x_data_normed = x_data / x_data_norms
    cov_acc = x_data_normed.T @ x_data_normed

    if sigma is not None and sigma > 0.:
        noise_mat = np.random.normal(size=(x_dims, x_dims), scale=sigma)
        i_lower = np.tril_indices(x_dims, -1)
        noise_mat[i_lower] = noise_mat.T[i_lower]  # symmetric noise
        cov_acc += noise_mat

    dp_cov = cov_acc / n_samples
    s, _, v = np.linalg.svd(dp_cov)
    sing_vecs = v.T[:, :pca_dims]

    return sing_vecs


# def pca_preprocess(x_data_train, x_data_test, sigma, d_enc):
#
#     x_data_train = x_data_train - x_data_train.mean()
#     x_data_train = x_data_train / np.abs(x_data_train).max()
#     x_data_test = x_data_test - x_data_test.mean()
#     x_data_test = x_data_test / np.abs(x_data_test).max()
#
#     sing_vecs = private_pca(x_data_train, sigma, d_enc)
#
#     x_train_pca = x_data_train @ sing_vecs  # (n,dx) (dimx, dimx') -> (bs,dimx')
#     x_test_pca = x_data_test @ sing_vecs
#     return x_train_pca, x_test_pca
#
#
# def pca_per_class(x_data, y_data, sigma, d_enc, n_labels):
#     s_vecs = []
#     for k in range(n_labels):
#         k_data = x_data[y_data == k]
#         k_data = k_data - k_data.mean()
#         k_data = k_data / np.abs(k_data).max()
#         s_vecs.append(private_pca(k_data, sigma, d_enc))
#     return np.stack(s_vecs, axis=0)  # (n_classes, d_in, d_proj)


def get_pca_projection(train_data_loader, sigma, pca_dims, device):
  x_data_train = []
  for x, _ in train_data_loader:
    x_data_train.append(pt.reshape(x, (x.shape[0], -1)))
  x_data_train = pt.cat(x_data_train, dim=0).cpu().numpy()

  x_data_train = x_data_train - x_data_train.mean()
  x_data_train = x_data_train / np.abs(x_data_train).max()

  sing_vecs = private_pca(x_data_train, sigma, pca_dims)
  sing_vecs = pt.tensor(sing_vecs, device=device)
  return sing_vecs


def train_single_release(gen, device, optimizer, epoch, rff_mmd_loss, log_interval, batch_size, n_data, pca_vecs):
  n_iter = n_data // batch_size
  for batch_idx in range(n_iter):
    gen_code, gen_labels = gen.get_code(batch_size, device)
    gen_data = gen(gen_code)
    if pca_vecs is not None:
      gen_data = apply_pca(pca_vecs, gen_data)
    loss = rff_mmd_loss(gen_data, gen_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * batch_size, n_data, loss.item()))


def compute_rff_loss(gen, data, labels, rff_mmd_loss, device, pca_vecs=None):
  bs = labels.shape[0]
  gen_code, gen_labels = gen.get_code(bs, device)
  gen_samples = gen(gen_code)
  if pca_vecs is not None:
    return rff_mmd_loss(apply_pca(pca_vecs, data), labels, apply_pca(pca_vecs, gen_samples), gen_labels)
  else:
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


def test(gen, device, test_loader, rff_mmd_loss, epoch, batch_size, log_dir, pca_vecs):
  test_loss = 0
  with pt.no_grad():
    for data, labels in test_loader:
      data, labels = data.to(device), labels.to(device)
      data = flat_data(data, labels, device, n_labels=10, add_label=False)
      loss = compute_rff_loss(gen, data, labels, rff_mmd_loss, device, pca_vecs=pca_vecs)
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
  parser.add_argument('--log-interval', type=int, default=10000, help='print updates after n steps')
  parser.add_argument('--base-log-dir', type=str, default='logs/gen/', help='path where logs for all runs are stored')
  parser.add_argument('--log-name', type=str, default=None, help='subdirectory for this run')
  parser.add_argument('--log-dir', type=str, default=None, help='override save path. constructed if None')
  parser.add_argument('--data', type=str, default='digits', help='options are digits and fashion')
  parser.add_argument('--synth-code_balanced', action='store_true', default=True, help='if true, make 60k synthetic code_balanced')

  # OPTIMIZATION
  parser.add_argument('--batch-size', '-bs', type=int, default=50)
  parser.add_argument('--test-batch-size', '-tbs', type=int, default=1000)
  parser.add_argument('--epochs', '-ep', type=int, default=10)
  parser.add_argument('--lr', '-lr', type=float, default=0.01, help='learning rate')
  parser.add_argument('--lr-decay', type=float, default=0.9, help='per epoch learning rate decay factor')

  # MODEL DEFINITION
  # parser.add_argument('--batch-norm', action='store_true', default=True, help='use batch norm in model')
  parser.add_argument('--d-code', '-dcode', type=int, default=5, help='random code dimensionality')
  parser.add_argument('--gen-spec', type=str, default='200', help='specifies hidden layers of generator')
  parser.add_argument('--kernel-sizes', '-ks', type=str, default='5,5', help='specifies conv gen kernel sizes')
  parser.add_argument('--n-channels', '-nc', type=str, default='16,8', help='specifies conv gen kernel sizes')
  parser.add_argument('--mmd-type', type=str, default='sphere', help='how to approx mmd', choices=['sphere', 'r+r'])

  # DP SPEC
  parser.add_argument('--d-rff', type=int, default=10000, help='number of random filters for apprixmate mmd')
  parser.add_argument('--rff-sigma', '-rffsig', type=str, default=None, help='standard dev. for filter sampling')
  parser.add_argument('--noise-factor', '-noise', type=float, default=5.0, help='privacy noise parameter')

  parser.add_argument('--reg-strength', type=float, default=1.0, help='')

  # PCA spec
  parser.add_argument('--pca', action='store_true', default=False, help='if true, use pca before rff encoding')
  parser.add_argument('--pca-dim', type=int, default=60, help='number of pca dimensions to keep')
  parser.add_argument('--pca-noise', type=float, default=.0, help='privacy noise for pca')

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
    ar.rff_sigma = '105' if ar.data == 'digits' else '127'


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
  n_data, n_feat = 60000, (784 if not ar.pca else ar.pca_dim)

  # load data
  train_loader, test_loader = get_mnist_dataloaders(ar.batch_size, ar.test_batch_size, use_cuda, dataset=ar.data)

  if ar.pca:
    pca_vecs = get_pca_projection(train_loader, ar.pca_noise, ar.pca_dim, device)
    print(f'pca dims: {pca_vecs.shape}')
  else:
    pca_vecs = None

  # init model
  gen = ConvCondGen(ar.d_code, ar.gen_spec, ar.n_labels, ar.n_channels, ar.kernel_sizes).to(device)

  # define loss function
  sr_loss, mb_loss, _ = get_rff_losses(train_loader, n_feat, ar.d_rff, ar.rff_sigma, device, ar.n_labels, ar.noise_factor,
                                       ar.mmd_type, pca_vecs)

  # init optimizer
  optimizer = pt.optim.Adam(list(gen.parameters()), lr=ar.lr)
  scheduler = StepLR(optimizer, step_size=1, gamma=ar.lr_decay)

  # training loop
  for epoch in range(1, ar.epochs + 1):
    train_single_release(gen, device, optimizer, epoch, sr_loss, ar.log_interval, ar.batch_size, n_data, pca_vecs)
    test(gen, device, test_loader, mb_loss, epoch, ar.batch_size, ar.log_dir, pca_vecs)
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
  # changes made to this file
  # 1. ensure batch_size % n_labels == 0
  # 2. generate (bs//n_labels) samples of each class every time
  # 3. define pairwise comparisons over batch
  # 4. add to mmd loss