# dp-hp

##Comparison between HP features ans RF features

Execute `error_comparison.py` to get the error comparison between HP features ans RF features (Figure 1)  and HP and random features comparison at a different length scale values (Figure 2).

##Experiments on image data

To run DP-HP experiments, use the following commands:

1. Creating the generated samples and predictive models evaluation

###MNIST

- ` python3 Me_sum_kernel_args.py --log-name *experiment name* --data digits -bs 200  --seed 0 --model_name FC -ep 10 --order-hermite 100  -kernel_length 0.005` for the non-private case

- ` python3 Me_sum_kernel_args.py --log-name *experiment name* --data digits -bs 200  --seed 0 --model_name FC -ep 10 --order-hermite 100  -kernel_length 0.005  --is-private True  --epsilon 1.0  --delta 1e-5` for $(1, 10^{-5})$-DP case

###FashionMNIST

- ` python3 Me_sum_kernel_args.py --log-name *experiment name* --data fashion -bs 200  --seed 0 --model_name CNN -ep 10 --order-hermite 100  -kernel_length 0.15` for the non-private case

- ` python3 Me_sum_kernel_args.py --log-name *experiment name* --data fashion -bs 200  --seed 0 --model_name CNN -ep 10 --order-hermite 100  -kernel_length 0.15  --is-private True  --epsilon 1.0  --delta 1e-5` for $(1, 10^{-5})$-DP case


2. Repoducing Fig. 4 and Fig. 5

-Run `plots_hp.py` that loads the results from different models from `mnist_results/` folder.

##Experiments on tabular data

1. Creating the generated samples and predictive models evaluation

- `python3 run_sum_kernel_tabular_data.py ` for non-private case

- `python3 run_sum_kernel_tabular_data.py  --is-private` for $(1, 10^{-5})$-DP case