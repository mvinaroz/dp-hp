from all_aux_files import test_results_subsampling_rate, test_gen_data, datasets_colletion_def
import os

seed=4
ep=10
hp=100
bs=200
lr=0.01
kernel=0.15

data='fashion'

log_dir='/Users/margaritavinaroz/Desktop/DPDR/dp_mehp/logs/gen/'
log_name='fashion_CNN_lr' + str(lr) + '_kernel' + str(kernel)+'-bs-' + str(bs)+'-seed-'+str(seed)+'-epochs-'+ str(ep)+ '-hp-' + str(hp)


skip_downstream_model=False
sampling_rate_synth=0.1

#dir_syn_data = log_dir + log_name+ '/'  + data +  '/synthetic_mnist.npz'

#data_tuple = datasets_colletion_def(syn_data, syn_labels,
#                                        data_pkg.train_data.data, data_pkg.train_data.targets,
#                                        data_pkg.test_data.data, data_pkg.test_data.targets)

test_results_subsampling_rate(data, log_name + '/' + data, log_dir + log_name, skip_downstream_model, sampling_rate_synth)
#final_score = test_gen_data(log_name + '/' + data, data, subsample=0.1, custom_keys='adaboost')    

