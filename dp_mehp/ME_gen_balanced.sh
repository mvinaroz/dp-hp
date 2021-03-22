#!/bin/bash
python3 ME_sum_kernel_args.py --log-name testing_v1_s0 -bs 1000 --sigma_arr 0.05 --test-batch-size 200 --data digits --seed 0 --model_name CNN -ep 1 -tbs 200

python3 ME_sum_kernel_args.py --log-name testing_v1_s1 -bs 1000 --sigma_arr 0.05 --test-batch-size 200 --data digits --seed 1 --model_name CNN -ep 1 -tbs 200

python3 ME_sum_kernel_args.py --log-name testing_v1_s2 -bs 1000 --sigma_arr 0.05 --test-batch-size 200 --data digits --seed 2 --model_name CNN -ep 1 -tbs 200

python3 ME_sum_kernel_args.py --log-name testing_v1_s3 -bs 1000 --sigma_arr 0.05 --test-batch-size 200 --data digits --seed 3 --model_name CNN -ep 1 -tbs 200

python3 ME_sum_kernel_args.py --log-name testing_v1_s4 -bs 1000 --sigma_arr 0.05 --test-batch-size 200 --data digits --seed 4 --model_name CNN -ep 1 -tbs 200