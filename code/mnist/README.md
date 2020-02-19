# Experiments on MNIST

In order to reproduce our experiments, you can run the commands outlined below.
All hyperparameters have been set to the values used in the paper and can be examined in the respective files.
Please note, that DP-MERF downloads datasets, while the DP-CGAN code assumes they already exist, so make sure to run DP-MERF first.

#### privacy analysis

All privacy settings in the scripts below are set for (9.6, 10^-5)-DP by default. Parameters for different privacy settings can be computed with 
`python3 dp_analysis.py`, after changing the parameters defined in that script.

## digit MNIST

#### DP-MERF
For the ($$2.9$$,10^-5)-DP model, append -noise 0.X and for (1.3,10^-5)-DP, append -noise 0.X  
- `python3 train_dp_autoencoder_directly.py --log-name dp_merf_digits --data digits`

#### DP-MERF+AE
first autoencoder training, then generator training
- `python3 train_dp_autoencoder.py --log-name dp_merf_ae_digits --data digits`
- `python3 train_dp_generator.py --ae-load-dir logs/ae/dp_merf_ae_digits/ --log-name dp_merf_ae_digits --data digits`

#### DP-CGAN
- `python3 dp_cgan_reference.py --data-save-str dpcgan_digits --data digits`

#### Evaluation
- `python3 synth_data_benchmark.py --data-log-name *experiment name* --data digits`

## fashion MNIST

All experiments are run with the same hyperparameters. The only change requires is switching the `--data` flag to `fashion`

#### DP-MERF
- `python3 train_dp_autoencoder_directly.py --log-name dp_merf_fashion --data fashion`
- `python3 synth_data_benchmark.py --data-log-name dp_merf_fashion --data fashion`

#### DP-MERF+AE
- `python3 train_dp_autoencoder.py --log-name dp_merf_ae_fashion --data fashion`
- `python3 train_dp_generator.py --ae-load-dir logs/ae/dp_merf_ae_fashion/ --log-name dp_merf_ae_fashion --data fashion`
- `python3 synth_data_benchmark.py --data-log-name dp_merf_ae_fashion --data fashion`

#### DP-CGAN
- `python3 dp_cgan_reference.py --data-save-str dpcgan_fashion --data fashion`
- `python3 synth_data_benchmark.py --data-log-name dpcgan_fashion --data fashion`