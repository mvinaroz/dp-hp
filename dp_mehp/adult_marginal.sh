#!/bin/bash

python3 adult_marginal_test.py --n_features 500  --iterations 2000 --batch_size 1000 --lr 0.001 --epsilon 1.0 --dataset simple --kernel gaussian --kernel_length 4 --d_hid 200 

python3 adult_marginal_test.py --n_features 1000  --iterations 2000 --batch_size 1000 --lr 0.001 --epsilon 1.0 --dataset simple --kernel gaussian --kernel_length 4 --d_hid 200 