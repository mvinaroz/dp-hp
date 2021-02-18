#!/bin/bash
python gen_balanced.py --data 2d -noise 0. --synth-spec-string norm_k5_n50000_row5_col5_noise0.2 -ep 50 --log-name jan21_test1 --gen-spec 200,500,500,200 --rff-sigma 0.3 -lr 1e-3 --d-rff 400 --loss-type hermite --sampling-rate 0.7
