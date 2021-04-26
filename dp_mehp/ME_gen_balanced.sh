#!/bin/bash
python3 Me_sum_kernel_args.py --log-name testing_v1_s0 -bs 200 --data digits --seed 0 --model_name FC -ep 1 --order-hermite 10 --heuristic-sigma --separate-kernel-length
