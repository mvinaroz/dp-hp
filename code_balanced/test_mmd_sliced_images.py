import torch as pt 
import torchvision
import numpy as np
from data_loading import get_dataloaders
from models_gen import FCCondGen, ConvCondGen
from mmd_real import get_real_mmd_loss
from torch.optim.lr_scheduler import StepLR
from aux import flatten_features, meddistance
from gen_balanced import log_gen_data, test_results
from gen_balanced import synthesize_data_with_uniform_labels
import os
from synth_data_benchmark import test_gen_data, test_passed_gen_data, datasets_colletion_def

#import sys
#sys.path.append('/Users/margaritavinaroz/Desktop/DPDR/using_pruned_net')
#from mmd_sum_kernel import mmd_loss

def check_slices_list(n_features, num_slices):
    num_pix=np.sqrt(n_features)
    assert num_pix % num_slice == 0, "select a number of patch that divides the number of pixels (28)"
    sliced_features=num_slice**2
    
    #List of the slices(patches) for each image.
    list_patch=[sliced_features*x  for x in range(int(n_features/sliced_features) + 1)]
    return list_patch

def mmd_per_class(x, y, sigma2, batch_size):
 """
 size(x) = mini_batch by feature_dimension
 size(y) = mini_batch by feature_dimension
 dist_xx = mini_batch by feature_dimension
 dist_yy = mini_batch by feature_dimension
 dist_xy = mini_batch by mini_batch by feature_dimension
 """

 m,feat_dim = x.shape
 n = y.shape[0]

 xx = pt.einsum('id,jd -> ijd', x, x) # m by m by feature_dimension
 yy = pt.einsum('id,jd -> ijd', y, y) # n by n by feature_dimension
 xy = pt.einsum('id,jd -> ijd', x, y) #  m by n by feature_dimension

 x2 = x**2 # m by feat_dim
 x2_extra_dim1 = x2[:,None,:]
 x2_extra_dim2 = x2[None,:,:]
 y2 = y**2
 y2_extra_dim1 = y2[:,None,:]
 y2_extra_dim2 = y2[None,:,:]

 # first term: sum_d sum_i sum_j k(x_i, x_j)
 dist_xx = pt.abs(x2_extra_dim1.repeat(1,m,1) - 2.0*xx + x2_extra_dim2.repeat(m,1,1))
 kxx = pt.sum(pt.exp(-dist_xx/(2.0*sigma2**2)))/(batch_size**2)

 # second term: sum_d sum_i sum_j k(y_i, y_j)
 dist_yy = pt.abs(y2_extra_dim1.repeat(1,n,1) - 2.0*yy + y2_extra_dim2.repeat(n,1,1))
 kyy = pt.sum(pt.exp(-dist_yy/(2.0*sigma2**2)))/(batch_size**2)

 # third term: sum_d sum_i sum_j k(x_i, y_j)
 dist_xy = pt.abs(x2_extra_dim1.repeat(1,n,1) - 2.0*xy + y2_extra_dim2.repeat(m,1,1))
 kxy = pt.sum(pt.exp(-dist_xy/(2.0*sigma2**2)))/(batch_size*batch_size)

 mmd = kxx + kyy - 2.0*kxy

 return mmd


def mmd_sum_kernel_sliced(data_enc, data_labels, gen_enc, gen_labels, n_labels, sigma2, method, list_patch):
    # set gen labels to scalars from one-hot
    _, gen_labels = pt.max(gen_labels, dim=1)
    batch_size_data = data_enc.shape[0]
    batch_size_enc=gen_enc.shape[0]
    feature_dim = data_enc.shape[1]
#    print("Feature data dimension: ", feature_dim)
    
    mmd_sum = 0
    for idx in range(n_labels):
        idx_data_enc = data_enc[data_labels == idx]
        #print("Data shape: ", idx_data_enc.shape)
        idx_gen_enc = gen_enc[gen_labels == idx]
        #print("Generated data shape: ", idx_gen_enc.shape)
        if method=='sum_kernel_sliced':
#            print("We are gonna compute the mmd for the image slices")
            for i in range(len(list_patch)-1):  
                #print("Indices: ", [list_patch[i],list_patch[i+1]])
                sliced_idx_data_enc=idx_data_enc [:, list_patch[i]:list_patch[i+1]]
                #print("Sliced data shape: ", sliced_idx_data_enc.shape)
                sliced_idx_gen_enc=idx_gen_enc[:, list_patch[i]:list_patch[i+1]]
                #print("Sliced generated data shape: ", sliced_idx_data_enc.shape)
                mmd_sum += mmd_per_class(sliced_idx_data_enc, sliced_idx_gen_enc, pt.sqrt(sigma2), batch_size_data)
#                print("MMD sum: ", mmd_sum)
        else:
            pass
    
    return mmd_sum

method="sum_kernel_sliced"
model_name="CNN"
report_intermidiate_result = False
n_features=784
num_slice=7


""" Check if cuda is available """
device = pt.device("cuda:0" if pt.cuda.is_available() else "cpu")

""" Download and load the MNIST data """
batch_size=500
test_batch_size=1000
data_name='digits'
gen_spec='500,500'

data_pkg=get_dataloaders(data_name, batch_size, test_batch_size, use_cuda=device, normalize=True, synth_spec_string=None, test_split=None)


""" Define the generator (init model) """
d_code=5
n_epochs = 20
use_sigmoid = data_name in {'digits', 'fashion'}

if model_name=="FC":
    model = FCCondGen(d_code, gen_spec, n_features, data_pkg.n_labels, use_sigmoid=False, batch_norm=True).to(device) 
elif model_name=="CNN":
    model = ConvCondGen(d_code, gen_spec, data_pkg.n_labels, '16,8', '5,5', use_sigmoid=True, batch_norm=True).to(device)
        # with CNN, logistic accuracy 73 percent (20 epochs) for full MMD
    
""" Training """
# set the scale length
num_iter = data_pkg.n_data/batch_size
if method=='sum_kernel_sliced':
    
    #Generate the list containing the slices for each image.
    list_patch=check_slices_list(n_features, num_slice)
    
    sigma2_arr = np.zeros((np.int(num_iter), np.int(n_features/num_slice**2)))
    
    for batch_idx, (data, labels) in enumerate(data_pkg.train_loader):
        # print('batch idx', batch_idx)
        data, labels = data.to(device), labels.to(device)
        data = flatten_features(data) # minibatch by feature_dim
        data_numpy = data.detach().cpu().numpy()
        #print("data to numpy array: ", data_numpy)
        count_dim=0
        for i in range(len(list_patch)-1):
            sliced_data_numpy=data_numpy[:, list_patch[i]:list_patch[i+1]]

            med = meddistance(data_numpy[:,list_patch[i]:list_patch[i+1]])
            sigma2 = med ** 2            
            sigma2_arr[batch_idx, count_dim] = sigma2
            count_dim+=1
            
else:
    pass

sigma2 = pt.tensor(np.mean(sigma2_arr))
print('length scale', sigma2)
    

""" Init optimizer """
lr=0.01
lr_decay=0.9
optimizer = pt.optim.Adam(list(model.parameters()), lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=lr_decay)

base_dir = 'logs/gen/'
log_dir = base_dir + data_name + method + model_name + '/'
log_dir2 = data_name + method + model_name + '/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
            
for epoch in range(1, n_epochs + 1):
    for batch_idx, (data, labels) in enumerate(data_pkg.train_loader):
        data, labels = data.to(device), labels.to(device)
        gen_code, gen_labels = model.get_code(batch_size, device)
        gen_samples = model(gen_code) # batch_size by 784

        data = flatten_features(data)  
        loss = mmd_sum_kernel_sliced(data, labels, gen_samples, gen_labels, data_pkg.n_labels, sigma2, method, list_patch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # end for
        
        n_data = len(data_pkg.train_loader.dataset)
        print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), n_data, loss.item()))


        log_gen_data(model, device, epoch, data_pkg.n_labels, log_dir)
        scheduler.step()
        
        if report_intermidiate_result:
            """ now we save synthetic data and test them on logistic regression """
            syn_data, syn_labels = synthesize_data_with_uniform_labels(model, device, gen_batch_size=batch_size,
                                                                       n_data=data_pkg.n_data,
                                                                       n_labels=data_pkg.n_labels)

            dir_syn_data = log_dir + data_name + '/synthetic_mnist'
            if not os.path.exists(dir_syn_data):
                os.makedirs(dir_syn_data)

            np.savez(dir_syn_data, data=syn_data, labels=syn_labels)
            final_score = test_gen_data(log_dir2 + data_name, data_name, subsample=0.1, custom_keys='logistic_reg')
            print('on logistic regression, accuracy is', final_score)

        # end if
    # end for

#########################################################################3
""" Once we have a trained generator, we store synthetic data from it and test them on logistic regression """
syn_data, syn_labels = synthesize_data_with_uniform_labels(model, device, gen_batch_size=batch_size,
                                                               n_data=data_pkg.n_data,
                                                               n_labels=data_pkg.n_labels)

dir_syn_data = log_dir + data_name + '/synthetic_mnist'
if not os.path.exists(dir_syn_data):
    os.makedirs(dir_syn_data)

np.savez(dir_syn_data, data=syn_data, labels=syn_labels)
final_score = test_gen_data(log_dir2 + data_name, data_name, subsample=0.1, custom_keys='logistic_reg')
print('on logistic regression, accuracy is', final_score)





