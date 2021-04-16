import os

import os
import numpy as np


seed=1
ep=10
hp=100
bs=200
lr=0.01
kernel=0.0003
data='digits'

log_dir='/Users/margaritavinaroz/Desktop/DPDR/dp_mehp/logs/gen/digits/gen/'
folder='digits_FC_lr' + str(lr) + '_kernel' + str(kernel)+'-bs-' + str(bs)+'-seed-'+str(seed)+'-epochs-'+ str(ep)+ '-hp-' + str(hp)
#folder='digits_FC_lr' + str(lr) + '_kernel' + str(kernel)+'-bs-' + str(bs)+'-seed-'+str(seed)+'-epochs-'+ str(ep)+ '-hp-' + str(hp)
path_folder=log_dir + folder

extra_dir= '/' + data + '/synth_eval'

total_path=path_folder + extra_dir 
os.chdir(total_path)

contenido = os.listdir(total_path)


for i in contenido:
    print(i)
    b=np.load(i)
    acc=b['accuracies']
    print(acc[1])