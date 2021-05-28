# dp-hp


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

### Tabular data

`dp_mehp/run_sum_kernel_image_data.py` contains the code for the tabular experiments


### Image data 
`dp_mehp/Me_sum_kernel_args.py` contains code for mnist and fashionmnist data experiments. See `add_info/README.md` for instructions on how to run the experiments.
