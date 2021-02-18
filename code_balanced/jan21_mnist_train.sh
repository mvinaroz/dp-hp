#!/bin/bash
if [ $1 -eq 0 ] ; then
python3 gen_balanced.py --log-name apr6_sr_conv_sig0_d0 --single-release -bs 100 --d-rff 10000 --rff-sigma 105 --test-batch-size 200 --data digits --seed 0 --conv-gen -ks 5,5 -nc 16,8 -noise 0.0
fi ; if [ $1 -eq 1 ] ; then
python3 gen_balanced.py --log-name apr6_sr_conv_sig0_d1 --single-release -bs 100 --d-rff 10000 --rff-sigma 105 --test-batch-size 200 --data digits --seed 1 --conv-gen -ks 5,5 -nc 16,8 -noise 0.0
fi ; if [ $1 -eq 2 ] ; then
python3 gen_balanced.py --log-name apr6_sr_conv_sig0_d2 --single-release -bs 100 --d-rff 10000 --rff-sigma 105 --test-batch-size 200 --data digits --seed 2 --conv-gen -ks 5,5 -nc 16,8 -noise 0.0
fi ; if [ $1 -eq 3 ] ; then
python3 gen_balanced.py --log-name apr6_sr_conv_sig0_d3 --single-release -bs 100 --d-rff 10000 --rff-sigma 105 --test-batch-size 200 --data digits --seed 3 --conv-gen -ks 5,5 -nc 16,8 -noise 0.0
fi ; if [ $1 -eq 4 ] ; then
python3 gen_balanced.py --log-name apr6_sr_conv_sig0_d4 --single-release -bs 100 --d-rff 10000 --rff-sigma 105 --test-batch-size 200 --data digits --seed 5 --conv-gen -ks 5,5 -nc 16,8 -noise 0.0
fi ; if [ $1 -eq 5 ] ; then
python3 gen_balanced.py --log-name apr6_sr_conv_sig5_d0 --single-release -bs 100 --d-rff 10000 --rff-sigma 105 --test-batch-size 200 --data digits --seed 0 --conv-gen -ks 5,5 -nc 16,8 -noise 5.0
fi ; if [ $1 -eq 6 ] ; then
python3 gen_balanced.py --log-name apr6_sr_conv_sig5_d1 --single-release -bs 100 --d-rff 10000 --rff-sigma 105 --test-batch-size 200 --data digits --seed 1 --conv-gen -ks 5,5 -nc 16,8 -noise 5.0
fi ; if [ $1 -eq 7 ] ; then
python3 gen_balanced.py --log-name apr6_sr_conv_sig5_d2 --single-release -bs 100 --d-rff 10000 --rff-sigma 105 --test-batch-size 200 --data digits --seed 2 --conv-gen -ks 5,5 -nc 16,8 -noise 5.0
fi ; if [ $1 -eq 8 ] ; then
python3 gen_balanced.py --log-name apr6_sr_conv_sig5_d3 --single-release -bs 100 --d-rff 10000 --rff-sigma 105 --test-batch-size 200 --data digits --seed 3 --conv-gen -ks 5,5 -nc 16,8 -noise 5.0
fi ; if [ $1 -eq 9 ] ; then
python3 gen_balanced.py --log-name apr6_sr_conv_sig5_d4 --single-release -bs 100 --d-rff 10000 --rff-sigma 105 --test-batch-size 200 --data digits --seed 5 --conv-gen -ks 5,5 -nc 16,8 -noise 5.0
fi ; if [ $1 -eq 10 ] ; then
python3 gen_balanced.py --log-name apr6_sr_conv_sig25_d0 --single-release -bs 100 --d-rff 10000 --rff-sigma 105 --test-batch-size 200 --data digits --seed 0 --conv-gen -ks 5,5 -nc 16,8 -noise 25.0
fi ; if [ $1 -eq 11 ] ; then
python3 gen_balanced.py --log-name apr6_sr_conv_sig25_d1 --single-release -bs 100 --d-rff 10000 --rff-sigma 105 --test-batch-size 200 --data digits --seed 1 --conv-gen -ks 5,5 -nc 16,8 -noise 25.0
fi ; if [ $1 -eq 12 ] ; then
python3 gen_balanced.py --log-name apr6_sr_conv_sig25_d2 --single-release -bs 100 --d-rff 10000 --rff-sigma 105 --test-batch-size 200 --data digits --seed 2 --conv-gen -ks 5,5 -nc 16,8 -noise 25.0
fi ; if [ $1 -eq 13 ] ; then
python3 gen_balanced.py --log-name apr6_sr_conv_sig25_d3 --single-release -bs 100 --d-rff 10000 --rff-sigma 105 --test-batch-size 200 --data digits --seed 3 --conv-gen -ks 5,5 -nc 16,8 -noise 25.0
fi ; if [ $1 -eq 14 ] ; then
python3 gen_balanced.py --log-name apr6_sr_conv_sig25_d4 --single-release -bs 100 --d-rff 10000 --rff-sigma 105 --test-batch-size 200 --data digits --seed 5 --conv-gen -ks 5,5 -nc 16,8 -noise 25.0
fi ; if [ $1 -eq 15 ] ; then
python3 gen_balanced.py --log-name apr6_sr_conv_sig0_f0 --single-release -bs 100 --d-rff 10000 --rff-sigma 127 --test-batch-size 200 --data fashion --seed 0 --conv-gen -ks 5,5 -nc 16,8 -noise 0.0
fi ; if [ $1 -eq 16 ] ; then
python3 gen_balanced.py --log-name apr6_sr_conv_sig0_f1 --single-release -bs 100 --d-rff 10000 --rff-sigma 127 --test-batch-size 200 --data fashion --seed 1 --conv-gen -ks 5,5 -nc 16,8 -noise 0.0
fi ; if [ $1 -eq 17 ] ; then
python3 gen_balanced.py --log-name apr6_sr_conv_sig0_f2 --single-release -bs 100 --d-rff 10000 --rff-sigma 127 --test-batch-size 200 --data fashion --seed 2 --conv-gen -ks 5,5 -nc 16,8 -noise 0.0
fi ; if [ $1 -eq 18 ] ; then
python3 gen_balanced.py --log-name apr6_sr_conv_sig0_f3 --single-release -bs 100 --d-rff 10000 --rff-sigma 127 --test-batch-size 200 --data fashion --seed 3 --conv-gen -ks 5,5 -nc 16,8 -noise 0.0
fi ; if [ $1 -eq 19 ] ; then
python3 gen_balanced.py --log-name apr6_sr_conv_sig0_f4 --single-release -bs 100 --d-rff 10000 --rff-sigma 127 --test-batch-size 200 --data fashion --seed 5 --conv-gen -ks 5,5 -nc 16,8 -noise 0.0
fi ; if [ $1 -eq 20 ] ; then
python3 gen_balanced.py --log-name apr6_sr_conv_sig5_f0 --single-release -bs 100 --d-rff 10000 --rff-sigma 127 --test-batch-size 200 --data fashion --seed 0 --conv-gen -ks 5,5 -nc 16,8 -noise 5.0
fi ; if [ $1 -eq 21 ] ; then
python3 gen_balanced.py --log-name apr6_sr_conv_sig5_f1 --single-release -bs 100 --d-rff 10000 --rff-sigma 127 --test-batch-size 200 --data fashion --seed 1 --conv-gen -ks 5,5 -nc 16,8 -noise 5.0
fi ; if [ $1 -eq 22 ] ; then
python3 gen_balanced.py --log-name apr6_sr_conv_sig5_f2 --single-release -bs 100 --d-rff 10000 --rff-sigma 127 --test-batch-size 200 --data fashion --seed 2 --conv-gen -ks 5,5 -nc 16,8 -noise 5.0
fi ; if [ $1 -eq 23 ] ; then
python3 gen_balanced.py --log-name apr6_sr_conv_sig5_f3 --single-release -bs 100 --d-rff 10000 --rff-sigma 127 --test-batch-size 200 --data fashion --seed 3 --conv-gen -ks 5,5 -nc 16,8 -noise 5.0
fi ; if [ $1 -eq 24 ] ; then
python3 gen_balanced.py --log-name apr6_sr_conv_sig5_f4 --single-release -bs 100 --d-rff 10000 --rff-sigma 127 --test-batch-size 200 --data fashion --seed 5 --conv-gen -ks 5,5 -nc 16,8 -noise 5.0
fi ; if [ $1 -eq 25 ] ; then
python3 gen_balanced.py --log-name apr6_sr_conv_sig25_f0 --single-release -bs 100 --d-rff 10000 --rff-sigma 127 --test-batch-size 200 --data fashion --seed 0 --conv-gen -ks 5,5 -nc 16,8 -noise 25.0
fi ; if [ $1 -eq 26 ] ; then
python3 gen_balanced.py --log-name apr6_sr_conv_sig25_f1 --single-release -bs 100 --d-rff 10000 --rff-sigma 127 --test-batch-size 200 --data fashion --seed 1 --conv-gen -ks 5,5 -nc 16,8 -noise 25.0
fi ; if [ $1 -eq 27 ] ; then
python3 gen_balanced.py --log-name apr6_sr_conv_sig25_f2 --single-release -bs 100 --d-rff 10000 --rff-sigma 127 --test-batch-size 200 --data fashion --seed 2 --conv-gen -ks 5,5 -nc 16,8 -noise 25.0
fi ; if [ $1 -eq 28 ] ; then
python3 gen_balanced.py --log-name apr6_sr_conv_sig25_f3 --single-release -bs 100 --d-rff 10000 --rff-sigma 127 --test-batch-size 200 --data fashion --seed 3 --conv-gen -ks 5,5 -nc 16,8 -noise 25.0
fi ; if [ $1 -eq 29 ] ; then
python3 gen_balanced.py --log-name apr6_sr_conv_sig25_f4 --single-release -bs 100 --d-rff 10000 --rff-sigma 127 --test-batch-size 200 --data fashion --seed 5 --conv-gen -ks 5,5 -nc 16,8 -noise 25.0
fi
