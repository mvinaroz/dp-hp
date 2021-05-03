import numpy as np
import kernel as k
from all_aux_files import meddistance
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from all_aux_files import find_rho
import kernel as k

from all_aux_files import get_dataloaders, flatten_features
from scipy.spatial.distance import cdist
import torch

def RFF_Gauss(n_features, X, W):
    """ this is a Pytorch version of Wittawat's code for RFFKGauss"""
    # Fourier transform formula from
    # http://mathworld.wolfram.com/FourierTransformGaussian.html

    W = torch.Tensor(W)
    XWT = torch.mm(X, torch.t(W))
    Z1 = torch.cos(XWT)
    Z2 = torch.sin(XWT)

    Z = torch.cat((Z1, Z2),1) * torch.sqrt(2.0/torch.Tensor([n_features]))
    return Z


def phi_recursion(phi_k, phi_k_minus_1, rho, degree, x_in):
  if degree == 0:
    phi_0 = (1 - rho) ** (0.25) * (1 + rho) ** (0.25) * torch.exp(-rho / (1 + rho) * x_in ** 2)
    return phi_0
  elif degree == 1:
    phi_1 = np.sqrt(2 * rho) * x_in * phi_k
    return phi_1
  else:  # from degree ==2 (k=1 in the recursion formula)
    k = degree - 1
    first_term = np.sqrt(rho) / np.sqrt(2 * (k + 1)) * 2 * x_in * phi_k
    second_term = rho / np.sqrt(k * (k + 1)) * k * phi_k_minus_1
    phi_k_plus_one = first_term - second_term
    return phi_k_plus_one


def compute_phi(x_in, n_degrees, rho, device):
  first_dim = x_in.shape[0]
  batch_embedding = torch.empty(first_dim, n_degrees, dtype=torch.float32, device=device)
  # batch_embedding = torch.zeros(first_dim, n_degrees).to(device)
  phi_i_minus_one, phi_i_minus_two = None, None
  for degree in range(n_degrees):
    phi_i = phi_recursion(phi_i_minus_one, phi_i_minus_two, rho, degree, x_in.squeeze())
    batch_embedding[:, degree] = phi_i

    phi_i_minus_two = phi_i_minus_one
    phi_i_minus_one = phi_i

  return batch_embedding


def feature_map_HP(k, x, rho, device):
  # k: degree of polynomial
  # rho: a parameter (related to length parameter)
  # x: where to evaluate the function at

  eigen_vals = (1 - rho) * (rho ** torch.arange(0, k + 1))
  eigen_vals = eigen_vals.to(device)
  phi_x = compute_phi(x, k + 1, rho, device)

  return phi_x, eigen_vals


def ME_with_HP(x, order, rho, device, n_training_data):
  n_data, input_dim = x.shape

  # reshape x, such that x is a long vector
  x_flattened = x.view(-1)
  x_flattened = x_flattened[:, None]
  phi_x_axis_flattened, eigen_vals_axis_flattened = feature_map_HP(order, x_flattened, rho, device)
  phi_x = phi_x_axis_flattened.reshape(n_data, input_dim, order + 1)
  phi_x = phi_x.type(torch.float)

  return phi_x

def err(A,B):
    return torch.norm(A-B)



torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data='fashion'
batch_size=60000
test_batch_size=1000
  
"""Load data"""
data_pkg = get_dataloaders(data, batch_size, test_batch_size, use_cuda=device, normalize=False, synth_spec_string=None, test_split=None)


"""evaluate the kernel function"""
for batch_idx, (data, labels) in enumerate(data_pkg.train_loader):
    #     # print('batch idx', batch_idx)
    data, labels = data.to(device), labels.to(device)
    data = flatten_features(data)

n = 200
x=data[0:n, :].detach().cpu().numpy() #Number of samples in x.
x_prime=data[n:2*n, :].detach().cpu().numpy() #number of samples in x_prime.
input_dim = x.shape[1]
 
#We have to compare for all points in x and x_prime  

med = meddistance(np.concatenate((x,x_prime),axis=0))
sigma2 = med**2
print('median heuristic finds sigma2 as ', sigma2)

Gaussian_kernel = k.KGauss(sigma2=sigma2)
K = Gaussian_kernel(x, x_prime)

"""Error plot"""
# evaluate the kernel function
max_order = 200
n_ME_data=x.shape[0]
#n_RF_data=data_pkg.n_data/2
err_RF = np.zeros(max_order)
err_HP = np.zeros(max_order)

# sigma2_for_HP = 0.05 # this is for fashion

### these are with unnormalized mnist
sigma2_for_HP = 0.05
rho = find_rho(sigma2_for_HP, False)

for i in range(max_order):
    print('# of features', i+1)
    n_features = i + 1  # so the order is from 0 to 1001
    draws = n_features // 2
    W_freq = np.random.randn(draws, data_pkg.n_features) / np.sqrt(sigma2)
    emb1 = RFF_Gauss(n_features, torch.Tensor(x), W_freq)
    emb2 = RFF_Gauss(n_features, torch.Tensor(x_prime), W_freq)
    RF = torch.matmul(emb2, emb1.transpose(1, 0))
    err_RF[i] = err(torch.Tensor(K), RF)


    order = i + 1
    phi_1 = ME_with_HP(torch.Tensor(x), order, rho, device, n_ME_data)
    phi_2 = ME_with_HP(torch.Tensor(x_prime), order, rho, device, n_ME_data)

    # HP1  = np.zeros((n, n))
    # for i in np.arange(0, n):
    #     for j in np.arange(0,n):
    #         Kd = np.zeros(input_dim)
    #         for d in np.arange(0,input_dim):
    #             Kd[d] = torch.dot(phi_2[i,d,:], phi_1[j,d,:])
    #         # end for over d
    #         HP1[i,j] = np.sum(Kd)

    HP = np.zeros((x.shape[0], x_prime.shape[0]))
    for h in range(x.shape[0]):
        for j in range(x_prime.shape[0]):
            k_HP = torch.sum(torch.einsum('jk, jk-> jk', phi_1[h,:,:], phi_2[j,:,:]))
            # prod=np.multiply(phi_1[h, :, :], phi_2[j, :, :]) #Compute the pairs phi_c(x_d)*phi_c(x_d')
            # k_HP=prod.sum() #Sum all the elements to get HP approx k(x, x')
            #print(k_HP)
            HP[h,j]=k_HP/input_dim
            

    err_HP[i] = err(torch.Tensor(K), HP)
    print('error of RF:%f and error of HP:%f' %(err_RF[i], err_HP[i]))

"""Plot ans save the results"""
font = {'family': 'normal',
        'weight': 'bold',
        'size': 22}

matplotlib.rc('font', **font)

plt.figure()
plt.subplot(212)
plt.plot(np.arange(0, max_order), err_RF, 'o-', linewidth=3.0)
plt.plot(np.arange(0, max_order), np.min(err_RF)*np.ones(max_order), 'k--')
plt.title('error from RF approximation')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('number of random features')
plt.subplot(211)
plt.plot(np.arange(0, max_order), err_HP, 'o-', linewidth=3.0)
plt.plot(np.arange(0, max_order), np.min(err_RF) * np.ones(max_order), 'k--')
plt.yscale('log')
plt.xscale('log')
plt.title('error from HP approximation')
plt.xlabel('order of polynomials')
plt.show()
# plt.savefig("prueba1_error.png", format='png')
