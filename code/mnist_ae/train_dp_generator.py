import os
import torch as pt
from torch.optim.lr_scheduler import StepLR
import argparse
import numpy as np
from models_ae import FCEnc, FCDec
from models_gen import FCGen, FCLabelGen, FCCondGen
from aux import rff_gauss, get_mnist_dataloaders, plot_mnist_batch, meddistance, save_gen_labels, \
  log_args, parse_n_hid, flat_data


def train(enc, gen, device, train_loader, optimizer, epoch, rff_mmd_loss, log_interval, uniform_labels):

  gen.train()
  for batch_idx, (data, labels) in enumerate(train_loader):
    # print(pt.max(data), pt.min(data))
    data = flat_data(data.to(device), labels.to(device), device, n_labels=10, add_label=False)

    bs = labels.shape[0]
    one_hots = pt.zeros(bs, 10, device=device)
    one_hots.scatter_(1, labels.to(device)[:, None], 1)
    if uniform_labels:
      gen_code, gen_labels = gen.get_code(bs, device)
    else:
      gen_code = gen.get_code(bs, device)
      gen_labels = None

    optimizer.zero_grad()
    data_enc = enc(data)
    gen_out = gen(gen_code)

    if uniform_labels:
      gen_out = (gen_out, gen_labels)


    loss = rff_mmd_loss(data_enc, one_hots, gen_out)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))


def test(enc, dec, gen, device, test_loader, rff_mmd_loss, epoch, batch_size, uniform_labels):
  gen.eval()

  test_loss = 0
  with pt.no_grad():
    for data, labels in test_loader:
      data = data.to(device)
      data_enc = enc(data)
      bs = labels.shape[0]

      one_hots = pt.zeros(bs, 10, device=device)
      one_hots.scatter_(1, labels.to(device)[:, None], 1)

      if uniform_labels:
        gen_code, gen_labels = gen.get_code(bs, device)
      else:
        gen_code = gen.get_code(bs, device)

      gen_out = gen(gen_code)
      gen_samples = dec(gen_out[0]) if isinstance(gen_out, tuple) else dec(gen_out)

      if uniform_labels:
        gen_out = (gen_out, gen_labels)

      test_loss += rff_mmd_loss(data_enc, one_hots, gen_out).item()  # sum up batch loss
  test_loss /= (len(test_loader.dataset) / batch_size)

  data_enc_batch = data_enc.cpu().numpy()
  med_dist = meddistance(data_enc_batch)
  print(f'med distance for encodings is {med_dist}, heuristic suggests sigma={med_dist ** 2}')

  plot_samples = gen_samples[:100, ...].cpu().numpy()
  plot_mnist_batch(plot_samples, 10, 10, f'samples/gen_samples_ep{epoch}.png')
  if isinstance(gen_out, tuple):
    save_gen_labels(gen_out[1][:100, ...].cpu().numpy(), 10, 10, f'samples/gen_labels_ep{epoch}')
  # bs = plot_samples.shape[0]
  # lit_samples = plot_samples - np.min(np.reshape(plot_samples, (bs, -1)), axis=1)[:, None, None, None]
  # lit_samples = lit_samples / np.max(np.reshape(lit_samples, (bs, -1)), axis=1)[:, None, None, None]
  # print(np.max(lit_samples), np.min(lit_samples))
  # plot_mnist_batch(lit_samples, 10, 10, f'samples/gen_samples_bright_ep{epoch}.png')
  print('Test set: Average loss: {:.4f}'.format(test_loss))


def get_rff_mmd_loss(d_enc, d_rff, rff_sigma, device, do_gen_labels, n_labels, noise_factor, batch_size):
  assert d_rff % 2 == 0
  w_freq = pt.tensor(np.random.randn(d_rff // 2, d_enc) / np.sqrt(rff_sigma)).to(pt.float32).to(device)

  if not do_gen_labels:
    def mean_embedding(x):
      return pt.mean(rff_gauss(x, w_freq), dim=0)

    def rff_mmd_loss(data_enc, gen_out):
      data_emb = mean_embedding(data_enc)
      gen_emb = mean_embedding(gen_out)
      noise = pt.randn(d_rff, device=device) * (noise_factor / batch_size)
      return pt.sum((data_emb + noise - gen_emb) ** 2)
  else:
    def label_mean_embedding(data, labels):
      return pt.mean(pt.einsum('ki,kj->kij', [rff_gauss(data, w_freq), labels]), 0)

    def rff_mmd_loss(data_enc, labels, gen_enc, gen_labels):
      data_emb = label_mean_embedding(data_enc, labels)  # (d_rff, n_labels)
      gen_emb = label_mean_embedding(gen_enc, gen_labels)
      noise = pt.randn(d_rff, n_labels, device=device) * (noise_factor / batch_size)
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

  # OPTIMIZATION
  parser.add_argument('--batch-size', '-bs', type=int, default=200)
  parser.add_argument('--test-batch-size', '-tbs', type=int, default=1000)
  parser.add_argument('--epochs', '-ep', type=int, default=20)
  parser.add_argument('--lr', '-lr', type=float, default=0.001)
  parser.add_argument('--lr-decay', type=float, default=0.9)

  # MODEL DEFINITION
  # parser.add_argument('--conv-ae', action='store_true', default=False)
  parser.add_argument('--d-enc', '-denc', type=int, default=5)
  parser.add_argument('--d-code', '-dcode', type=int, default=5)
  parser.add_argument('--gen-hid', type=str, default='100,100')
  parser.add_argument('--gen-labels', action='store_true', default=False)
  parser.add_argument('--uniform-labels', action='store_true', default=False)

  # DP SPEC
  parser.add_argument('--d-rff', type=int, default=100)
  parser.add_argument('--rff-sigma', '-rffsig', type=int, default=90.0)
  parser.add_argument('--noise-factor', '-noise', type=float, default=0.0)

  # AE info
  # parser.add_argument('--conv-ae', action='store_true', default=False)
  parser.add_argument('--ae-label', action='store_true', default=False)
  parser.add_argument('--ae-ce-loss', action='store_true', default=False)
  parser.add_argument('--ae_clip', type=float, default=1e-5)
  parser.add_argument('--ae_noise', type=float, default=2.0)
  parser.add_argument('--ae-enc-hid', type=str, default='300,100')
  parser.add_argument('--ae-dec-hid', type=str, default='100,300')
  parser.add_argument('--ae-load-dir', type=str, default=None)

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
    log_dir = ar.base_log_dir + ar.log_name
  else:
    gen_type = f'{"uniform_" if ar.uniform_labels else ""}{"labeled_" if ar.gen_labels else "unlabeled_"}'
    gen_spec = f'd{ar.d_enc}_gen{ar.gen_hid}_sig{ar.noise_factor}_dcode{ar.d_code}_drff{ar.d_rff}_rffsig{ar.rff_sigma}'
    ae_type = f'{"label_" if ar.label_ae else ""}{"ce_" if ar.label_ae else "mse_"}'
    ae_spec = f'enc{ar.enc_hid}_dec{ar.dec_hid}_clip{ar.clip_norm}_sig{ar.noise_factor}'
    log_dir = ar.base_log_dir + gen_type + gen_spec + '_AE:' + ae_type + ae_spec
  return log_dir


def preprocess_args(args):
  if args.seed is None:
    args.seed = np.random.randint(0, 1000)

  if args.ae_load_dir is None:
    spec_str = f'd{args.d_enc}_enc{args.ae_enc_hid}_dec{args.ae_dec_hid}_clip{args.ae_clip}_sig{args.ae_noise}'
    type_str = f'{"label_" if args.ae_label else ""}{"ce_" if args.ae_ce_loss else "mse_"}'
    args.ae_load_dir = 'logs/ae/' + type_str + spec_str


def main():
  # Training settings

  ar = get_args()

  pt.manual_seed(ar.seed)

  use_cuda = not ar.no_cuda and pt.cuda.is_available()
  train_loader, test_loader = get_mnist_dataloaders(ar.batch_size, ar.test_batch_size, use_cuda)

  device = pt.device("cuda" if use_cuda else "cpu")

  d_data = 28**2+ar.n_labels if ar.ae_label else 28**2
  enc = FCEnc(d_data, parse_n_hid(ar.ae_enc_hid), ar.d_enc)
  dec = FCDec(ar.d_enc, parse_n_hid(ar.ae_dec_hid), d_data, use_sigmoid=ar.ae_ce_loss)
  enc.load_state_dict(pt.load(f'models/mnist_enc_fc_d_enc_{ar.d_enc}_clip_{ar.ae_clip}_noise_{ar.ae_noise}.pt'))
  dec_extended_dict = pt.load(f'models/mnist_dec_fc_d_enc_{ar.d_enc}_clip_{ar.ae_clip}_noise_{ar.ae_noise}.pt')
  dec_reduced_dict = {k: dec_extended_dict[k] for k in dec.state_dict().keys()}
  dec.load_state_dict(dec_reduced_dict)

  enc = enc.to(device)
  dec = dec.to(device)

  if ar.gen_labels:
    if ar.uniform_labels:
      gen = FCCondGen(ar.d_code, parse_n_hid(ar.gen_hid), ar.d_enc, ar.n_labels)
    else:
      gen = FCLabelGen(ar.d_code, parse_n_hid(ar.gen_hid), ar.d_enc, ar.n_labels)
  else:
    gen = FCGen(ar.d_code, parse_n_hid(ar.gen_hid), ar.d_enc)
  gen = gen.to(device)

  rff_mmd_loss = get_rff_mmd_loss(ar.d_enc, ar.d_rff, ar.rff_sigma, device, ar.gen_labels,
                                  ar.n_labels, ar.noise_factor, ar.batch_size)

  optimizer = pt.optim.Adam(list(gen.parameters()), lr=ar.lr)
  scheduler = StepLR(optimizer, step_size=1, gamma=ar.lr_decay)
  for epoch in range(1, ar.epochs + 1):
    train(enc, gen, device, train_loader, optimizer, epoch, rff_mmd_loss, ar.log_interval, ar.uniform_labels)
    test(enc, dec, gen, device, test_loader, rff_mmd_loss, epoch, ar.batch_size, ar.uniform_labels)
    scheduler.step()

  if ar.save_model:
    pt.save(gen.state_dict(), f'models/mnist_gen_fc_d_code_{ar.d_code}_d_enc_{ar.d_enc}_noise_{ar.noise_factor}.pt')


if __name__ == '__main__':
  main()
