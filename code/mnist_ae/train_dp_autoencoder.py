import os
import numpy as np
import torch as pt
import torch.nn.functional as nnf
from torch.optim.lr_scheduler import StepLR
import argparse
from collections import namedtuple
from backpack import extend, backpack
from backpack.extensions import BatchGrad, BatchL2Grad
from models_ae import FCEnc, FCDec, ConvEnc, ConvDec
from aux import get_mnist_dataloaders, plot_mnist_batch, save_gen_labels, log_args, flat_data, expand_vector


def train(enc, dec, device, train_loader, optimizer, epoch, losses, dp_spec, label_ae, conv_ae, log_interval):
  enc.train()
  dec.train()
  for batch_idx, (data, labels) in enumerate(train_loader):
    data = data.to(device)
    labels = labels.to(device)
    if not conv_ae:
      data = flat_data(data, labels, device, add_label=label_ae)

    optimizer.zero_grad()

    data_enc = enc(data)
    reconstruction = dec(data_enc)

    v = bin_ce_loss(reconstruction, data) if losses.do_ce else mse_loss(reconstruction, data)
    if losses.wsiam > 0.:
      v = v + losses.wsiam * siamese_loss(data_enc, labels, losses.msiam, per_sample=True)
    l_enc = loss_to_backpack_mse(v, losses.l_enc)

    if dp_spec.clip is None:
      l_enc.backward()
      bp_global_norms, rec_loss, siam_loss = None, None, None
    else:
      l_enc.backward(retain_graph=True)  # get grads for encoder

      # wipe grads from decoder:
      for param in dec.parameters():
        param.grad = None

      reconstruction = dec(data_enc.detach())

      rec_loss = bin_ce_loss(reconstruction, data) if losses.do_ce else mse_loss(reconstruction, data)
      if losses.wsiam > 0.:
        siam_loss = losses.wsiam * siamese_loss(data_enc, labels, losses.msiam, per_sample=True)
        full_loss = rec_loss + siam_loss
      else:
        siam_loss = None
        full_loss = rec_loss
      l_dec = loss_to_backpack_mse(full_loss, losses.l_dec)

      with backpack(BatchGrad(), BatchL2Grad()):
        l_dec.backward()

      # compute global gradient norm from parameter gradient norms
      bp_squared_param_norms = [p.batch_l2**2 for p in dec.parameters()]
      bp_global_norms = pt.sqrt(pt.sum(pt.stack(bp_squared_param_norms), dim=0))
      # manual_squared_param_norms = [pt.norm(pt.reshape(p.grad_batch, (p.grad_batch.shape[0], -1)), dim=1) ** 2
      #                               for p in dec.parameters()]
      # manual_global_norms = pt.sqrt(pt.sum(pt.stack(manual_squared_param_norms), dim=0))
      # manual_squared_param_norms1 = [pt.sum(pt.reshape(p.grad_batch, (p.grad_batch.shape[0], -1))**2, dim=1)
      #                               for p in dec.parameters()]
      # manual_global_norms1 = pt.sqrt(pt.sum(pt.stack(manual_squared_param_norms1), dim=0))
      # print(f'backpack_global_norm: {bp_global_norms[:3]}')
      # print(f'manual_global_norm: {manual_global_norms[:3]}')
      # print(f'manual_global_norm1: {manual_global_norms1[:3]}')
      # for param in dec.parameters():
      #     bp_norm = param.batch_l2
      #     bp_grad = param.grad_batch
      #     manual_norm_squared = pt.norm(pt.reshape(bp_grad, (bp_grad.shape[0], -1)), dim=1)**2
      #     print(f'norm_comp: {bp_norm[0]}, {manual_norm_squared[0]}')

      clips = pt.clamp_max(dp_spec.clip / bp_global_norms, 1.)
      # aggregate samplewise grads, replace normal grad
      for param in dec.parameters():
        # print(param.batch_l2, param.grad_batch)
        clipped_sample_grads = param.grad_batch * expand_vector(clips, param.grad_batch)
        clipped_grad = pt.mean(clipped_sample_grads, dim=0)

        if dp_spec.noise is not None:
          bs = clipped_grad.shape[0]
          noise_sdev = (2 * dp_spec.noise * dp_spec.clip / bs)
          clipped_grad = clipped_grad + pt.rand_like(clipped_grad, device=device) * noise_sdev
        param.grad = clipped_grad

    optimizer.step()
    if batch_idx % log_interval == 0:
      if dp_spec.clip is not None:
        print(f'max_norm:{pt.max(bp_global_norms).item()}, mean_norm:{pt.mean(bp_global_norms).item()}')

      n_done = batch_idx * len(data)
      n_data = len(train_loader.dataset)
      frac_done = 100. * batch_idx / len(train_loader)
      if siam_loss is None:
        loss_str = 'Loss: {:.6f}'.format(l_enc.item())
      else:
        l_s, l_r = pt.mean(siam_loss).item(), pt.mean(rec_loss).item()
        loss_str = 'Loss: full {:.6f}, rec {:.6f}, siam {:.6f}'.format(l_enc.item(), l_r, l_s)
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\t{}'.format(epoch, n_done, n_data, frac_done, loss_str))


def test(enc, dec, device, test_loader, epoch, losses, label_ae, conv_ae, log_dir):
  enc.eval()
  dec.eval()

  rec_loss_agg = 0
  siam_loss_agg = 0
  with pt.no_grad():
    for data, labels in test_loader:
      data = data.to(device)
      labels = labels.to(device)
      if not conv_ae:
        data = flat_data(data, labels, device, add_label=label_ae)

      data_enc = enc(data)
      reconstruction = dec(data_enc)
      rec_loss = bin_ce_loss(reconstruction, data) if losses.do_ce else mse_loss(reconstruction, data)
      rec_loss_agg += pt.sum(rec_loss).item()
      if losses.wsiam > 0.:
        siam_loss = losses.wsiam * siamese_loss(data_enc, labels, losses.msiam, per_sample=True)
        siam_loss_agg += pt.sum(siam_loss).item()

  n_data = len(test_loader.dataset)
  rec_loss_agg /= n_data
  siam_loss_agg /= n_data
  full_loss = rec_loss_agg + siam_loss_agg

  if label_ae:
    rec_labels = reconstruction[:100, 784:].cpu().numpy()
    save_gen_labels(rec_labels, 10, 10, log_dir + f'rec_ep{epoch}_labels', save_raw=False)
    reconstruction = reconstruction[:100, :784].reshape(100, 28, 28).cpu().numpy()
  else:
    reconstruction = reconstruction[:100, ...].cpu().numpy()

  plot_mnist_batch(reconstruction, 10, 10, log_dir + f'rec_ep{epoch}', denorm=not losses.do_ce)

  print('Test set: Average loss: full {:.4f}, rec {}, siam {}'.format(full_loss, rec_loss_agg, siam_loss_agg))


def mse_loss(reconstruction, data):  # this could be done directly with backpack but this way is more consistent
  mse = (reconstruction - data)**2
  return pt.sum(pt.reshape(mse, (mse.shape[0], -1)), dim=1, keepdim=True)


def bin_ce_loss(reconstruction, data):
  bce = nnf.binary_cross_entropy(reconstruction, data, reduction='none')
  return pt.sum(pt.reshape(bce, (bce.shape[0], -1)), dim=1, keepdim=True)


def siamese_loss(feats, labels, margin, per_sample=False):
    """
    :param feats: (bs, nfeats)
    :param labels: (bs)
    :param margin: ()
    :return: scalar loss
    """
    bs = feats.shape[0]
    assert bs % 2 == 0  # bs is even
    split = bs // 2
    feats_a = feats[:split, :]
    labels_a = labels[:split]
    feats_b = feats[split:, :]
    labels_b = labels[split:]

    # L =  Y * d^2 + (1-Y) * max(margin - d^2, 0)
    match = pt.eq(labels_a, labels_b).float()
    no_match = 1. - match
    dist = pt.sqrt(pt.sum((feats_a - feats_b) ** 2, dim=1))
    loss = match * dist + no_match * nnf.relu(margin - dist)
    if not per_sample:
      return pt.mean(loss)
    else:
      return 0.5 * pt.cat([loss, loss])[:, None]


def loss_to_backpack_mse(loss_vector, pt_mse_loss):
  return pt_mse_loss(pt.sqrt(loss_vector), pt.zeros_like(loss_vector))


def get_args():
  parser = argparse.ArgumentParser()

  # BASICS
  parser.add_argument('--no-cuda', action='store_true', default=False)
  parser.add_argument('--seed', type=int, default=None)  # sampled if none
  parser.add_argument('--log-interval', type=int, default=100)
  parser.add_argument('--base-log-dir', type=str, default='logs/ae/')
  parser.add_argument('--log-name', type=str, default=None)  # set for custom save subdir
  parser.add_argument('--log-dir', type=str, default=None)  # constructed if None (only set thisto completely alter loc)
  parser.add_argument('--n-labels', type=int, default=10)

  # OPTIMIZATION
  parser.add_argument('--batch-size', '-bs', type=int, default=200)
  parser.add_argument('--test-batch-size', '-tbs', type=int, default=1000)
  parser.add_argument('--epochs', '-ep', type=int, default=10)
  parser.add_argument('--lr', type=float, default=0.001)
  parser.add_argument('--lr-decay', type=float, default=0.9)
  parser.add_argument('--siam-loss-weight', '-wsiam', type=float, default=0.)
  parser.add_argument('--siam-loss-margin', '-msiam', type=float, default=1.)

  # MODEL DEFINITION
  parser.add_argument('--d-enc', '-denc', type=int, default=5)
  parser.add_argument('--conv-ae', action='store_true', default=False)
  parser.add_argument('--ce-loss', action='store_true', default=False)
  parser.add_argument('--label-ae', action='store_true', default=False)
  # parser.add_argument('--enc-spec', type=str, default='300,100')
  # parser.add_argument('--dec-spec', type=str, default='100,300')
  # parser.add_argument('--enc-spec', type=str, default='1,8,16,16')
  # parser.add_argument('--dec-spec', type=str, default='16,16,8,1')
  parser.add_argument('--enc-spec', type=str, default='1,4,8,8')
  parser.add_argument('--dec-spec', type=str, default='8,8,4,1')

  # DP SPEC
  parser.add_argument('--clip-norm', '-clip', type=float, default=1e-5)
  parser.add_argument('--noise-factor', '-noise', type=float, default=0.0)
  parser.add_argument('--clip-per-layer', action='store_true', default=False)  # not used yet

  ar = parser.parse_args()

  # HACKS FOR QUICK ACCESS
  # ar.ce_loss = True
  # ar.conv_ae = True
  # ar.label_ae = True
  # ar.siam_loss_weight = 10.
  # ar.siam_loss_margin = 10.

  if ar.log_dir is None:
    ar.log_dir = get_log_dir(ar)
  if not os.path.exists(ar.log_dir):
    os.makedirs(ar.log_dir)
  preprocess_args(ar)
  log_args(ar.log_dir, ar)
  return ar


def get_log_dir(args):
  if args.log_name is not None:
    log_dir = args.base_log_dir + args.log_name
  else:
    type_str = f'{"label_" if args.label_ae else ""}{"ce" if args.ce_loss else "mse"}_'
    spec_str = f'd{args.d_enc}_enc{args.enc_spec}_dec{args.dec_spec}_clip{args.clip_norm}_sig{args.noise_factor}'
    siam_str = f'_siam_w{args.siam_loss_weight}_m{args.siam_loss_margin}' if args.siam_loss_weight > 0. else ''
    log_dir = args.base_log_dir + type_str + spec_str + siam_str + '/'
  return log_dir


def preprocess_args(args):
  if args.seed is None:
    args.seed = np.random.randint(0, 1000)

  assert not (args.conv_ae and args.label_ae)  # not supported yet, will try conv with siamese loss first


def main():
  # Training settings
  ar = get_args()
  assert not ar.clip_per_layer  # not implemented yet
  pt.manual_seed(ar.seed)

  dp_spec = namedtuple('dp_spec', ['clip', 'noise', 'per_layer'])(ar.clip_norm, ar.noise_factor, ar.clip_per_layer)

  use_cuda = not ar.no_cuda and pt.cuda.is_available()
  train_loader, test_loader = get_mnist_dataloaders(ar.batch_size, ar.test_batch_size,
                                                    use_cuda, normalize=not ar.ce_loss)

  device = pt.device("cuda" if use_cuda else "cpu")
  d_data = 28**2+ar.n_labels if ar.label_ae else 28**2

  enc_spec = tuple([int(k) for k in ar.enc_spec.split(',')])
  dec_spec = tuple([int(k) for k in ar.dec_spec.split(',')])

  if ar.conv_ae:
    enc = ConvEnc(ar.d_enc, enc_spec, extra_conv=True).to(device)
    dec = extend(ConvDec(ar.d_enc, dec_spec, extra_conv=True, use_sigmoid=ar.ce_loss)).to(device)
    # print(list(enc.layers[0].parameters()), list(enc.parameters()))
  else:
    enc = FCEnc(d_in=d_data, d_hid=enc_spec, d_enc=ar.d_enc).to(device)
    dec = extend(FCDec(d_enc=ar.d_enc, d_hid=dec_spec, d_out=d_data, use_sigmoid=ar.ce_loss).to(device))

  loss_nt = namedtuple('losses', ['l_enc', 'l_dec', 'do_ce', 'wsiam', 'msiam'])
  losses = loss_nt(pt.nn.MSELoss(), extend(pt.nn.MSELoss()), ar.ce_loss, ar.siam_loss_weight, ar.siam_loss_margin)
  optimizer = pt.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=ar.lr)

  scheduler = StepLR(optimizer, step_size=1, gamma=ar.lr_decay)
  for epoch in range(1, ar.epochs + 1):
    train(enc, dec, device, train_loader, optimizer, epoch, losses, dp_spec, ar.label_ae, ar.conv_ae, ar.log_interval)
    test(enc, dec, device, test_loader, epoch, losses, ar.label_ae, ar.conv_ae, ar.log_dir)
    scheduler.step()

  pt.save(enc.state_dict(), ar.log_dir + 'enc.pt')
  pt.save(dec.state_dict(), ar.log_dir + 'dec.pt')


if __name__ == '__main__':
  main()
