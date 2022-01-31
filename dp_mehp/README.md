# dp-hp

## Comparison between HP features ans RF features

Execute `error_comparison.py` to get the error comparison between HP features ans RF features (Figure 1)  and HP and random features comparison at a different length scale values (Figure 1).

## 2D data (Fig 2)

Run `dp_mehp_synth_data_2d.py`

## Experiments on image data

To run DP-HP experiments, use the following commands:

1. Creating the generated samples and predictive models evaluation

### MNIST

- ` python3 prod_and_sum_kernel_image.py --log-name *experiment name* --data digits -bs 200  --seed 0 --model-name FC -ep 10  -lr 0.01 --order-hermite-sum 100 --order-hermite-prod 20 --kernel-length-sum 0.005 --kernel-length-prod 0.005 --gamma 5 --prod-dimension 2` for the non-private case

- ` python3 prod_and_sum_kernel_image.py --log-name *experiment name* --data digits -bs 200  --seed 0  --ep 10 --lr 0.01 --order-hermite-sum 100 --order-hermite-prod 20 --model-name FC --kernel-length-sum 0.005 --kernel-length-prod 0.005 --gamma 20 --prod-dimension 2 --split --split-sum-ratio 0.8 --is-private` for $(1, 10^{-5})$-DP case

### FashionMNIST

- ` python3 prod_and_sum_kernel_image.py--log-name *experiment name* --data fashion -bs 200  --seed 0 --model-name CNN -ep 10  -lr 0.01 --order-hermite-sum 100 --order-hermite-prod 20 --kernel-length-sum 0.15 --kernel-length-prod 0.15 --gamma 20 --prod-dimension 2` for the non-private case

- ` python3 prod_and_sum_kernel_image.py --log-name *experiment name* --data fashion -bs 200  --seed 0 --model-name CNN -ep 10  -lr 0.01 --order-hermite-sum 100 --order-hermite-prod 20 --kernel-length-sum 0.15 --kernel-length-prod 0.15 --gamma 10 --prod-dimension 2  --split --split-sum-ratio 0.8 --is-private` for $(1, 10^{-5})$-DP case


2. Repoducing Fig. 3 and Fig. 6

-Run `code_balanced/plot_results.py` that loads the results from different models from `code_balanced/plots/` folder.

## Experiments on tabular data

1. Results in table 1. are obtained with `discretized_datasets.py`

2. Results in table 2. are obatained with `run_sum_prod_kernel_tabular_data.py`
