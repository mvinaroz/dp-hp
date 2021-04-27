#!/bin/bash
python3 Me_sum_kernel_args.py --log-name digits_FC_lr0.01_kernel0.0004-bs-100-seed-1-epochs-20-hp-100 -bs 100 --data digits --seed 1 --model_name FC -ep 20 --order-hermite 100 --kernel-length 0.0004

python3 Me_sum_kernel_args.py --log-name digits_FC_lr0.01_kernel0.0004-bs-200-seed-1-epochs-20-hp-100 -bs 200 --data digits --seed 1 --model_name FC -ep 20 --order-hermite 100 --kernel-length 0.0004

python3 Me_sum_kernel_args.py --log-name digits_FC_lr0.01_kernel0.0004-bs-500-seed-1-epochs-20-hp-100 -bs 500 --data digits --seed 1 --model_name FC -ep 20 --order-hermite 100 --kernel-length 0.0004

python3 Me_sum_kernel_args.py --log-name digits_FC_lr0.01_kernel0.0004-bs-1000-seed-1-epochs-20-hp-100 -bs 1000 --data digits --seed 1 --model_name FC -ep 20 --order-hermite 100 --kernel-length 0.0004

python3 Me_sum_kernel_args.py --log-name digits_FC_lr0.01_kernel0.0004-bs-2000-seed-1-epochs-20-hp-100 -bs 2000 --data digits --seed 1 --model_name FC -ep 20 --order-hermite 100 --kernel-length 0.0004
