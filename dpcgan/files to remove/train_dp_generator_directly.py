import os
import torch as pt
from torch.optim.lr_scheduler import StepLR
import argparse
import numpy as np
from models_gen import FCGen, FCLabelGen, FCCondGen, FCGenBig
from aux import plot_mnist_batch, meddistance, save_gen_labels, log_args, flat_data
from data_loading import get_mnist_dataloaders
from rff_mmd_approx import rff_sphere
from real_mmd_loss import mmd_g, get_squared_dist
from autodp import rdp_acct, rdp_bank


def train(gen, device, train_loader, optimizer, epoch, rff_mmd_loss, log_interval, do_gen_labels, uniform_labels):

  for batch_idx, (data, labels) in enumerate(train_loader):
    # print(pt.max(data), pt.min(data))
    data = flat_data(data.to(device), labels.to(device), device, n_labels=10, add_label=False)

    bs = labels.shape[0]
    if not do_gen_labels:
      loss = rff_mmd_loss(data, gen(gen.get_code(bs, device)))

    elif uniform_labels:
      one_hots = pt.zeros(bs, 10, device=device)
      one_hots.scatter_(1, labels.to(device)[:, None], 1)
      gen_code, gen_labels = gen.get_code(bs, device)
      loss = rff_mmd_loss(data, one_hots, gen(gen_code), gen_labels)
    else:
      one_hots = pt.zeros(bs, 10, device=device)
      one_hots.scatter_(1, labels.to(device)[:, None], 1)
      gen_enc, gen_labels = gen(gen.get_code(bs, device))
      loss = rff_mmd_loss(data, one_hots, gen_enc, gen_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))


def test(gen, device, test_loader, rff_mmd_loss, epoch, batch_size, do_gen_labels, uniform_labels, log_dir):
  test_loss = 0
  gen_labels, ordered_labels = None, None
  with pt.no_grad():
    for data, labels in test_loader:
      data = data.to(device)
      data = flat_data(data.to(device), labels.to(device), device, n_labels=10, add_label=False)

      bs = labels.shape[0]

      if not do_gen_labels:
        gen_samples = gen(gen.get_code(bs, device))
        gen_labels = None
        loss = rff_mmd_loss(data, gen_samples)

      elif uniform_labels:
        one_hots = pt.zeros(bs, 10, device=device)
        one_hots.scatter_(1, labels.to(device)[:, None], 1)
        gen_code, gen_labels = gen.get_code(bs, device)
        gen_samples = gen(gen_code)
        loss = rff_mmd_loss(data, one_hots, gen_samples, gen_labels)

      else:
        one_hots = pt.zeros(bs, 10, device=device)
        one_hots.scatter_(1, labels.to(device)[:, None], 1)
        gen_samples, gen_labels = gen(gen.get_code(bs, device))
        loss = rff_mmd_loss(data, one_hots, gen_samples, gen_labels)

      test_loss += loss.item()  # sum up batch loss
  test_loss /= (len(test_loader.dataset) / batch_size)

  data_enc_batch = data.cpu().numpy()
  med_dist = meddistance(data_enc_batch)
  print(f'med distance for encodings is {med_dist}, heuristic suggests sigma={med_dist ** 2}')

  if uniform_labels:
    ordered_labels = pt.repeat_interleave(pt.arange(10), 10)[:, None].to(device)
    gen_code, gen_labels = gen.get_code(100, device, labels=ordered_labels)
    gen_samples = gen(gen_code).detach()

  plot_samples = gen_samples[:100, ...].cpu().numpy()
  plot_mnist_batch(plot_samples, 10, 10, log_dir + f'samples_ep{epoch}')
  if gen_labels is not None and ordered_labels is None:
    save_gen_labels(gen_labels[:100, ...].cpu().numpy(), 10, 10, log_dir + f'labels_ep{epoch}')
  # bs = plot_samples.shape[0]
  # lit_samples = plot_samples - np.min(np.reshape(plot_samples, (bs, -1)), axis=1)[:, None, None, None]
  # lit_samples = lit_samples / np.max(np.reshape(lit_samples, (bs, -1)), axis=1)[:, None, None, None]
  # print(np.max(lit_samples), np.min(lit_samples))
  # plot_mnist_batch(lit_samples, 10, 10, f'samples/gen_samples_bright_ep{epoch}.png')
  print('Test set: Average loss: {:.4f}'.format(test_loss))


def get_rff_mmd_loss(d_enc, d_rff, rff_sigma, device, do_gen_labels, n_labels, noise_factor, batch_size, real_mmd):
  assert d_rff % 2 == 0
  w_freq = pt.tensor(np.random.randn(d_rff // 2, d_enc) / np.sqrt(rff_sigma)).to(pt.float32).to(device)
  if real_mmd:
    print('we use full MMD here')
    def rff_mmd_loss(data_enc, gen_enc):
      dxx, dxy, dyy = get_squared_dist(data_enc, gen_enc)
      return mmd_g(dxx, dxy, dyy, batch_size, sigma=np.sqrt(rff_sigma))

  elif not do_gen_labels:
    def mean_embedding(x):
      return pt.mean(rff_sphere(x, w_freq), dim=0)

    def rff_mmd_loss(data_enc, gen_out):
      data_emb = mean_embedding(data_enc)
      gen_emb = mean_embedding(gen_out)
      noise = pt.randn(d_rff, device=device) * (2 * noise_factor / batch_size)
      return pt.sum((data_emb + noise - gen_emb) ** 2)
  else:
    print('we use the random feature mean embeddings here')
    def label_mean_embedding(data, labels):
      return pt.mean(pt.einsum('ki,kj->kij', [rff_sphere(data, w_freq), labels]), 0)

    def rff_mmd_loss(data_enc, labels, gen_enc, gen_labels):
      data_emb = label_mean_embedding(data_enc, labels)  # (d_rff, n_labels)
      gen_emb = label_mean_embedding(gen_enc, gen_labels)
      noise = pt.randn(d_rff, n_labels, device=device) * (2 * noise_factor / batch_size)
      return pt.sum((data_emb + noise - gen_emb) ** 2)
  return rff_mmd_loss




def get_args():
  parser = argparse.ArgumentParser()

  # BASICS
  parser.add_argument('--no-cuda', action='store_true', default=False)
  parser.add_argument('--seed', type=int, default=None)
  parser.add_argument('--n-labels', type=int, default=10)
  parser.add_argument('--log-interval', type=int, default=100)
  parser.add_argument('--base-log-dir', type=str, default='logs/gen/')
  parser.add_argument('--log-name', type=str, default=None)
  parser.add_argument('--log-dir', type=str, default=None)  # constructed if None
  parser.add_argument('--synth-mnist', action='store_true', default=False)
  parser.add_argument('--data', type=str, default='digits')  # options are digits and fashion

  # OPTIMIZATION
  parser.add_argument('--batch-size', '-bs', type=int, default=500)
  parser.add_argument('--test-batch-size', '-tbs', type=int, default=1000)
  parser.add_argument('--epochs', '-ep', type=int, default=5)
  parser.add_argument('--lr', '-lr', type=float, default=0.01)
  parser.add_argument('--lr-decay', type=float, default=0.9)

  # MODEL DEFINITION
  # parser.add_argument('--conv-ae', action='store_true', default=False)
  parser.add_argument('--d-code', '-dcode', type=int, default=5)
  parser.add_argument('--gen-spec', type=str, default='100,50')
  parser.add_argument('--gen-labels', action='store_true', default=True)
  parser.add_argument('--uniform-labels', action='store_true', default=True)
  parser.add_argument('--big-gen', action='store_true', default=False)
  parser.add_argument('--batch-norm', action='store_true', default=True)
  parser.add_argument('--real-mmd', action='store_true', default=False)

  # DP SPEC
  parser.add_argument('--d-rff', type=int, default=1000)
  parser.add_argument('--rff-sigma', '-rffsig', type=float, default=100.0)
  parser.add_argument('--noise-factor', '-noise', type=float, default=0.4)
  ar = parser.parse_args()

  # HACKS FOR QUICK ACCESS
  # ar.ae_label = True
  # ar.ae_ce_loss = True
  # ar.ae_conv = True
  # ar.gen_labels = True
  # ar.uniform_labels = True

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
    gen_type = f'{"uniform_" if ar.uniform_labels else ""}{"labeled_" if ar.gen_labels else "unlabeled_"}'
    print('gen_type', gen_type)
    gen_spec = f'gen{ar.gen_spec}_sig{ar.noise_factor}_dcode{ar.d_code}_drff{ar.d_rff}_rffsig{ar.rff_sigma}_bs{ar.batch_size}'
    print('gen_spec', gen_spec)
    log_dir = ar.base_log_dir + gen_type + gen_spec + '/'
  return log_dir


def preprocess_args(args):
  if args.seed is None:
    args.seed = np.random.randint(0, 1000)

  assert args.data in {'digits', 'fashion'}
  assert args.gen_labels or not args.uniform_labels
  assert not (args.gen_labels and args.real_mmd)



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
  train_loader, test_loader = get_mnist_dataloaders(ar.batch_size, ar.test_batch_size, use_cuda, normalize=False,
                                                    dataset=ar.data)

  device = pt.device("cuda" if use_cuda else "cpu")

  gen_spec = tuple([int(k) for k in ar.gen_spec.split(',')]) if ar.gen_spec is not None else None
  if ar.gen_labels:
    if ar.uniform_labels:
      gen = FCCondGen(ar.d_code, gen_spec, 784, ar.n_labels, use_sigmoid=True)
    else:
      gen = FCLabelGen(ar.d_code, gen_spec, 784, ar.n_labels, use_sigmoid=True)
  else:
    if ar.big_gen:
      gen = FCGenBig(ar.d_code, gen_spec, 784, use_sigmoid=True)
    else:
      gen = FCGen(ar.d_code, gen_spec, 784, use_sigmoid=True, batch_norm=ar.batch_norm)
  gen = gen.to(device)

  rff_mmd_loss = get_rff_mmd_loss(784, ar.d_rff, ar.rff_sigma, device, ar.gen_labels,
                                  ar.n_labels, ar.noise_factor, ar.batch_size, ar.real_mmd)



  ### compute the final epsilon dp based on the input arguments

  # (1) privacy parameters for four types of Gaussian mechanisms
  sigma = ar.noise_factor

  # (2) desired delta level
  delta = 1e-5

  # (5) number of training steps
  batch_size = ar.batch_size
  n_epochs = ar.epochs
  n_data = 60000
  steps_per_epoch = n_data // n_data
  n_steps = steps_per_epoch * n_epochs

  # (6) sampling rate
  prob = batch_size / n_data

  """ end of input arguments """

  """ now use autodp to calculate the cumulative privacy loss """
  # declare the moment accountants
  acct = rdp_acct.anaRDPacct()

  eps_seq = []
  print_every_n = 100
  for i in range(1, n_steps+1):
      acct.compose_subsampled_mechanism(lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x), prob)
      eps_seq.append(acct.get_eps(delta))
      if i % print_every_n == 0 or i == n_steps:
          print("[", i, "]Privacy loss is", (eps_seq[-1]))

  print("Composition of subsampled Gaussian mechanisms gives ", (acct.get_eps(delta), delta))

  optimizer = pt.optim.Adam(list(gen.parameters()), lr=ar.lr)
  scheduler = StepLR(optimizer, step_size=1, gamma=ar.lr_decay)
  for epoch in range(1, ar.epochs + 1):
    train(gen, device, train_loader, optimizer, epoch, rff_mmd_loss, ar.log_interval,
          ar.gen_labels, ar.uniform_labels)
    test(gen, device, test_loader, rff_mmd_loss, epoch, ar.batch_size,
         ar.gen_labels, ar.uniform_labels, ar.log_dir)
    scheduler.step()

  pt.save(gen.state_dict(), ar.log_dir + 'gen.pt')
  if ar.synth_mnist:
    assert ar.uniform_labels
    syn_data, syn_labels = synthesize_mnist_with_uniform_labels(gen, device)
    np.savez(ar.log_dir + 'synthetic_mnist', data=syn_data, labels=syn_labels)


if __name__ == '__main__':
  main()