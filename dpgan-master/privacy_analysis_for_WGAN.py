### privacy analysis for DP WGAN using autodp
""" Important Assumptions """
### (1) each generator for each class is trained separately
### (2) each training has the same epoch, same mini-batch size



from autodp import privacy_calibrator

#########################################
"""" modify these for your dataset """
#########################################
total_num_datapt = 50000
mini_batch_size = 64
num_epochs = 1000
num_classes = 2 # how many classes do you consider?  how many generators do you plan to train?
d_iters = 5 # number of iterations per discriminator update
#########################################

# Privacy level we set to use
eps = 1.0
delta = 1e-5

# input arguments to autodp
gamma = mini_batch_size / total_num_datapt # sampling probability
steps_per_epoch = total_num_datapt // mini_batch_size
n_steps = steps_per_epoch * num_epochs * num_classes * d_iters

print('gamma', gamma)
print('steps_per_epoch', steps_per_epoch)
print('n_steps', n_steps)

# Using RDP
params = privacy_calibrator.gaussian_mech(eps,delta,prob=gamma,k=n_steps)

this_is_the_sigma_you_want_to_use_for_each_training = params['sigma']

print(f'eps,delta,gamma = ({eps},{delta},{gamma}) ==> Noise level sigma=',this_is_the_sigma_you_want_to_use_for_each_training)
