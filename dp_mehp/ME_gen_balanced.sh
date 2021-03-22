#!/bin/bash
python3 ME_sum_kernel_args.py --log-name testing_v1_s0 -bs 1000 --sigma_arr 0.05 --test-batch-size 200 --data digits --seed 0 --model_name CNN -ep 1 -tbs 200
