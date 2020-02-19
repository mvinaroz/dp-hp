# dp-merf


### Dependencies
Versions numbers are based on our system and may not need to be exact matches. 

    python 3.6
    torch 1.3.1              
    torchvision 0.4.2
    numpy 1.16.4
    scipy 1.3.1
    pandas 1.0.1
    scikit-learn 0.21.2
    matplotlib 3.1.0 (plotting)
    seaborn 0.10.0 (more plotting)
    sdgym 0.1.0 (handling tabular datasets)
    autodp 0.1 (privacy analysis)
    backpack-for-pytorch 1.0.1 (efficient DP-SGD for DP-MERF+AE)
    tensorboardX 1.7 (some logging)
    tensorflow-gpu 1.14.0 (DP-CGAN)


## Repository Structure

###Tabular data

`code/single_generator_priv_all.py` contains the code for the tabular experiments


### High-dimensional data
`code/mnist` contains code for mnist data experiments. See `code/mnist/README.md` for instructions on how to run the experiments.
- `code/mnist/train_dp_generator_directly.py` trains a DP-MERF generator and can create a synthetic dataset afterwards.
- `code/mnist/train_dp_autoencoder.py` trains the autoencoder for a DP-MERF+AE model.
- `code/mnist/train_dp_generator.py` trains the generator for a DP-MERF+AE model and can create a synthetic dataset afterwards.
- `code/mnist/synth_data_benchmark.py` evaluates synthetic datasets on classification tasks
- `code/mnist/dp_analysis.py` computes privacy loss for training DP-MERF models
- `code/mnist/dp_cgan_reference.py` trains CGAN generator and generates synthetic dataset.
- remaining files contain helper functions

