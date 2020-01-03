import torch

def Gaussian_RF(sigma2, n_features, X, gpu_mode):
    mean_emb = RFF_Gauss(sigma2, n_features, X, gpu_mode)
    return mean_emb

def distance_RF(mean_emb1, mean_emb2):
    mean_emb1_avg = torch.mean(mean_emb1,0)
    mean_emb2_avg = torch.mean(mean_emb2,0)
    distance_RF_eval = torch.dist(mean_emb1_avg, mean_emb2_avg, p=2)**2
    return distance_RF_eval

def RFF_Gauss(sigma2, n_features, X, gpu_mode):
    """ I converted the numpy code RFFKGauss (W. Jitkrittum wrote) into Pytorch """
    # Fourier transform formula from
    # http://mathworld.wolfram.com/FourierTransformGaussian.html

    dim_input_2 = X.size(2)**2
    X = X.view(-1, dim_input_2)# reshape X

    n, d = X.size()
    draws = n_features // 2

    sigma2 = torch.Tensor([sigma2])
    W = torch.randn(draws, d) / torch.sqrt(sigma2)
    if gpu_mode:
        W = W.cuda()

    # n x draws
    # XWT = X.dot(W.T)
    XWT = torch.mm(X, torch.t(W))
    Z1 = torch.cos(XWT)
    Z2 = torch.sin(XWT)

    # n_features = torch.Tensor([n_features])
    # Z = torch.hstack((Z1, Z2)) * torch.sqrt(1.0 / n_features)
    scaling = torch.sqrt(2.0/torch.Tensor([n_features]))
    if gpu_mode:
        scaling = scaling.cuda()
    Z = torch.cat((Z1, Z2),1) * scaling
    return Z