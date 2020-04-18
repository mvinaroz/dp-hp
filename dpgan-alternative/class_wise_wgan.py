# original source https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan/wgan.py

import argparse
import os
import numpy as np

import torch as pt
from torchvision.utils import save_image
from wgan_models import Generator, Discriminator
from data_loading import get_single_label_dataloader
from backpack import extend, backpack
from backpack.extensions import BatchGrad, BatchL2Grad


def parse_arguments():

  parser = argparse.ArgumentParser()
  parser.add_argument("--n-epochs", type=int, default=200, help="number of epochs of training")
  parser.add_argument("--batch-size", type=int, default=64, help="size of the batches")
  parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
  parser.add_argument("--n-cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
  parser.add_argument("--latent-dim", type=int, default=100, help="dimensionality of the latent space")
  parser.add_argument("--img-size", type=int, default=28, help="size of each image dimension")
  parser.add_argument("--channels", type=int, default=1, help="number of image channels")
  parser.add_argument("--n-critic", type=int, default=5, help="number of training steps for discriminator per iter")
  parser.add_argument("--clip-value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
  parser.add_argument("--sample-interval", type=int, default=1000, help="interval betwen image samples")
  parser.add_argument("--print-interval", type=int, default=50, help="interval betwen image samples")

  parser.add_argument("--seed", type=int, default=42, help="random seed")
  parser.add_argument("--log-name", type=str, default='test', help="name of folder where results are stored")
  parser.add_argument('--overwrite', action='store_true', default=False, help='only write to existing log-name if true')
  parser.add_argument('--synth-data', action='store_true', default=True, help='make synthetic data if true')

  parser.add_argument('--dp-clip', '-clip', type=float, default=0.01, help='samplewise gradient L2 clip norm')
  parser.add_argument('--dp-noise', '-noise', type=float, default=0.0, help='DP-SGD noise parameter')

  return parser.parse_args()


def make_log_dirs(ar, n_labels=10):
  if ar.synth_data:
    os.makedirs(f"synth_data/{ar.log_name}", exist_ok=ar.overwrite or ar.log_name == 'test')

  for l in range(n_labels):
    os.makedirs(f"run_logs/{ar.log_name}/l{l}", exist_ok=ar.overwrite or ar.log_name == 'test')


def log_args(log_dir, args):
  """ print and save all args """
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  with open(os.path.join(log_dir, 'args_log'), 'w') as f:
    lines = [' • {:<25}- {}\n'.format(key, val) for key, val in vars(args).items()]
    f.writelines(lines)
    for line in lines:
      print(line.rstrip())
  print('-------------------------------------------')


def train_batch(real_imgs, device, dis_opt, gen_opt, dis, gen, clip_value, train_gen, dp_clip, dp_noise):
  real_imgs = real_imgs.to(device)  # Configure input
  dis_opt.zero_grad()

  z = gen.get_noise(real_imgs.shape[0], device)  # Sample noise as generator input
  fake_imgs = gen(z).detach()  # Generate a batch of images

  # loss_d = -pt.mean(dis(real_imgs) - dis(fake_imgs))  # Adversarial loss - reformulated

  if dp_noise == 0.:
    loss_d = -pt.mean(dis(real_imgs)) + pt.mean(dis(fake_imgs))  # Adversarial loss
    loss_d.backward()
    norms_real, clips_real, norms_fake, clips_fake = None, None, None, None
    ld = loss_d.item()
  else:
    norms_real, clips_real, norms_fake, clips_fake, ld = dp_sgd_backward(dis, real_imgs, fake_imgs,
                                                                         device, dp_clip, dp_noise, dis_opt)

  dis_opt.step()


  # Clip weights of discriminator
  for p in dis.parameters():
    p.data.clamp_(-clip_value, clip_value)

  if train_gen:  # Train the generator every n_critic iterations
    gen_opt.zero_grad()
    gen_imgs = gen(z)
    loss_g = -pt.mean(dis(gen_imgs))  # Adversarial loss
    loss_g.backward()
    gen_opt.step()
    lg = loss_g.item()
  else:
    lg = 0.

  return ld, lg, fake_imgs, norms_real, clips_real, norms_fake, clips_fake


def dp_sgd_backward(dis, real_imgs, fake_imgs, device, clip_norm, noise_factor, dis_opt):
  """
  since only one part of the loss depends on the data, we compute its gradients separately and noise them up
  in order to maintain similar gradient sizes, gradients for the other loss are clipped to the same per sample norm
  """
  # real data loss first:
  params = list(dis.parameters())
  loss_real = -pt.mean(dis(real_imgs))
  with backpack(BatchGrad(), BatchL2Grad()):
    loss_real.backward(retain_graph=True)

  squared_param_norms_real = [p.batch_l2 for p in params]  # first we get all the squared parameter norms...
  global_norms_real = pt.sqrt(pt.sum(pt.stack(squared_param_norms_real), dim=0))  # ...then compute the global norms...
  global_clips_real = pt.clamp_max(clip_norm / global_norms_real, 1.)  # ...and finally get a vector of clipping factors

  perturbed_grads = []
  for idx, param in enumerate(params):
    clipped_sample_grads = param.grad_batch * expand_vector(global_clips_real, param.grad_batch)
    clipped_grad = pt.sum(clipped_sample_grads, dim=0)  # after clipping we sum over the batch

    noise_sdev = noise_factor * 2 * clip_norm  # gaussian noise standard dev is computed (sensitivity is 2*clip)...
    perturbed_grad = clipped_grad + pt.randn_like(clipped_grad, device=device) * noise_sdev  # ...and applied
    perturbed_grads.append(perturbed_grad)  # store perturbed grads

  dis_opt.zero_grad()
  # now add fake data loss gradients:
  loss_fake = pt.mean(dis(fake_imgs))
  with backpack(BatchGrad(), BatchL2Grad()):
    loss_fake.backward()

  squared_param_norms_fake = [p.batch_l2 for p in params]  # first we get all the squared parameter norms...
  global_norms_fake = pt.sqrt(pt.sum(pt.stack(squared_param_norms_fake), dim=0))  # ...then compute the global norms...
  global_clips_fake = pt.clamp_max(clip_norm / global_norms_fake, 1.)  # ...and finally get a vector of clipping factors

  for idx, param in enumerate(params):
    clipped_sample_grads = param.grad_batch * expand_vector(global_clips_fake, param.grad_batch)
    clipped_grad = pt.sum(clipped_sample_grads, dim=0)  # after clipping we sum over the batch

    param.grad = clipped_grad + perturbed_grads[idx]

  ld = loss_real.item() + loss_fake.item()
  return global_norms_real, global_clips_real, global_norms_fake, global_clips_fake, ld


def expand_vector(vec, tgt_tensor):
  tgt_shape = [vec.shape[0]] + [1] * (len(tgt_tensor.shape) - 1)
  return vec.view(*tgt_shape)


def log_progress(batches_done, ep_len, ld, lg, epoch, gen_imgs, ar, label, is_final_batch):
  if batches_done % ar.print_interval == 0:
    ep_frac = batches_done % ep_len

    print(f'[Epoch {epoch}/{ar.n_epochs}] [Batch {ep_frac}/{ep_len}] [D loss: {ld}] [G loss: {lg}]')

  if batches_done % ar.sample_interval == 0 or is_final_batch:
    save_image(gen_imgs.data[:25], f"run_logs/{ar.log_name}/l{label}/{batches_done}.png", nrow=5, normalize=True)


def make_synth_data(gen, n_data, device, log_name, label, batch_size=300):
  samples = []
  while n_data > 0:
    if n_data < batch_size:
      batch_size = n_data

    samples.append(gen(gen.get_noise(batch_size, device)).detach().cpu().numpy())
    n_data -= batch_size
  samples = np.concatenate(samples)
  np.save(f'synth_data/{log_name}/synth_l{label}.npy', samples)


def combine_synth_data(labels, log_name):
  samples = []
  for label in labels:
    samples.append(np.load(f'synth_data/{log_name}/synth_l{label}.npy'))
  samples = np.concatenate(samples)
  samples = samples[np.random.permutation(samples.shape[0])]  # mix em up
  np.save(f'synth_data/{log_name}/synth_data_full.npy', samples)


def train_model_for_label(ar, label):
  img_shape = (ar.channels, ar.img_size, ar.img_size)
  device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
  gen = Generator(ar.latent_dim, img_shape).to(device)  # Initialize generator and discriminator
  dis = Discriminator(img_shape).to(device)
  if ar.dp_noise > 0.:
    dis = extend(dis)

  dataloader, n_data = get_single_label_dataloader(ar.batch_size, label)

  # Optimizers
  gen_opt = pt.optim.RMSprop(gen.parameters(), lr=ar.lr)
  dis_opt = pt.optim.RMSprop(dis.parameters(), lr=ar.lr)

  batches_done = 0
  for epoch in range(ar.n_epochs):
    for idx, (real_imgs, _) in enumerate(dataloader):
      train_gen = batches_done % ar.n_critic == 0
      is_final_batch = epoch + 1 == ar.n_epochs and idx + 1 == len(dataloader)

      ret_vals = train_batch(real_imgs, device, dis_opt, gen_opt, dis, gen, ar.clip_value, train_gen,
                             ar.dp_clip, ar.dp_noise)
      ld, lg, fake_imgs, norms_real, clips_real, norms_fake, clips_fake = ret_vals
      log_progress(batches_done, len(dataloader), ld, lg, epoch, fake_imgs, ar, label, is_final_batch)
      batches_done += 1
  if ar.synth_data:
    make_synth_data(gen, n_data, device, ar.log_name, label)


def main():

  ar = parse_arguments()
  make_log_dirs(ar)
  pt.manual_seed(ar.seed)
  log_args(f"synth_data/{ar.log_name}/", ar)

  # labels = list(range(5, 7))
  labels = list(range(10))
  for label in labels:
    train_model_for_label(ar, label)

  if ar.synth_data:
    combine_synth_data(labels, ar.log_name)


if __name__ == '__main__':
  main()
