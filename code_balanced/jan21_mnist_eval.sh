#!/bin/bash
if [ $1 -eq 0 ] ; then
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.1 --log-results --data-log-name apr6_sr_conv_sig0_d0 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.1 --log-results --data-log-name apr6_sr_conv_sig0_d1 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.1 --log-results --data-log-name apr6_sr_conv_sig0_d2 --data digits
fi ; if [ $1 -eq 1 ] ; then
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.1 --log-results --data-log-name apr6_sr_conv_sig0_d3 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.1 --log-results --data-log-name apr6_sr_conv_sig0_d4 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.1 --log-results --data-log-name apr6_sr_conv_sig5_d0 --data digits
fi ; if [ $1 -eq 2 ] ; then
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.1 --log-results --data-log-name apr6_sr_conv_sig5_d1 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.1 --log-results --data-log-name apr6_sr_conv_sig5_d2 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.1 --log-results --data-log-name apr6_sr_conv_sig5_d3 --data digits
fi ; if [ $1 -eq 3 ] ; then
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.1 --log-results --data-log-name apr6_sr_conv_sig5_d4 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.1 --log-results --data-log-name apr6_sr_conv_sig25_d0 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.1 --log-results --data-log-name apr6_sr_conv_sig25_d1 --data digits
fi ; if [ $1 -eq 4 ] ; then
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.1 --log-results --data-log-name apr6_sr_conv_sig25_d2 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.1 --log-results --data-log-name apr6_sr_conv_sig25_d3 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.1 --log-results --data-log-name apr6_sr_conv_sig25_d4 --data digits
fi ; if [ $1 -eq 5 ] ; then
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.1 --log-results --data-log-name apr6_sr_conv_sig0_f0 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.1 --log-results --data-log-name apr6_sr_conv_sig0_f1 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.1 --log-results --data-log-name apr6_sr_conv_sig0_f2 --data fashion
fi ; if [ $1 -eq 6 ] ; then
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.1 --log-results --data-log-name apr6_sr_conv_sig0_f3 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.1 --log-results --data-log-name apr6_sr_conv_sig0_f4 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.1 --log-results --data-log-name apr6_sr_conv_sig5_f0 --data fashion
fi ; if [ $1 -eq 7 ] ; then
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.1 --log-results --data-log-name apr6_sr_conv_sig5_f1 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.1 --log-results --data-log-name apr6_sr_conv_sig5_f2 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.1 --log-results --data-log-name apr6_sr_conv_sig5_f3 --data fashion
fi ; if [ $1 -eq 8 ] ; then
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.1 --log-results --data-log-name apr6_sr_conv_sig5_f4 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.1 --log-results --data-log-name apr6_sr_conv_sig25_f0 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.1 --log-results --data-log-name apr6_sr_conv_sig25_f1 --data fashion
fi ; if [ $1 -eq 9 ] ; then
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.1 --log-results --data-log-name apr6_sr_conv_sig25_f2 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.1 --log-results --data-log-name apr6_sr_conv_sig25_f3 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.1 --log-results --data-log-name apr6_sr_conv_sig25_f4 --data fashion
fi ; if [ $1 -eq 10 ] ; then
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.01 --log-results --data-log-name apr6_sr_conv_sig0_d0 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.01 --log-results --data-log-name apr6_sr_conv_sig0_d1 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.01 --log-results --data-log-name apr6_sr_conv_sig0_d2 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.01 --log-results --data-log-name apr6_sr_conv_sig0_d3 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.01 --log-results --data-log-name apr6_sr_conv_sig0_d4 --data digits
fi ; if [ $1 -eq 11 ] ; then
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.01 --log-results --data-log-name apr6_sr_conv_sig5_d0 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.01 --log-results --data-log-name apr6_sr_conv_sig5_d1 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.01 --log-results --data-log-name apr6_sr_conv_sig5_d2 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.01 --log-results --data-log-name apr6_sr_conv_sig5_d3 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.01 --log-results --data-log-name apr6_sr_conv_sig5_d4 --data digits
fi ; if [ $1 -eq 12 ] ; then
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.01 --log-results --data-log-name apr6_sr_conv_sig25_d0 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.01 --log-results --data-log-name apr6_sr_conv_sig25_d1 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.01 --log-results --data-log-name apr6_sr_conv_sig25_d2 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.01 --log-results --data-log-name apr6_sr_conv_sig25_d3 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.01 --log-results --data-log-name apr6_sr_conv_sig25_d4 --data digits
fi ; if [ $1 -eq 13 ] ; then
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.01 --log-results --data-log-name apr6_sr_conv_sig0_f0 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.01 --log-results --data-log-name apr6_sr_conv_sig0_f1 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.01 --log-results --data-log-name apr6_sr_conv_sig0_f2 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.01 --log-results --data-log-name apr6_sr_conv_sig0_f3 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.01 --log-results --data-log-name apr6_sr_conv_sig0_f4 --data fashion
fi ; if [ $1 -eq 14 ] ; then
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.01 --log-results --data-log-name apr6_sr_conv_sig5_f0 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.01 --log-results --data-log-name apr6_sr_conv_sig5_f1 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.01 --log-results --data-log-name apr6_sr_conv_sig5_f2 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.01 --log-results --data-log-name apr6_sr_conv_sig5_f3 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.01 --log-results --data-log-name apr6_sr_conv_sig5_f4 --data fashion
fi ; if [ $1 -eq 15 ] ; then
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.01 --log-results --data-log-name apr6_sr_conv_sig25_f0 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.01 --log-results --data-log-name apr6_sr_conv_sig25_f1 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.01 --log-results --data-log-name apr6_sr_conv_sig25_f2 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.01 --log-results --data-log-name apr6_sr_conv_sig25_f3 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.01 --log-results --data-log-name apr6_sr_conv_sig25_f4 --data fashion
fi ; if [ $1 -eq 16 ] ; then
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.001 --log-results --data-log-name apr6_sr_conv_sig0_d0 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.001 --log-results --data-log-name apr6_sr_conv_sig0_d1 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.001 --log-results --data-log-name apr6_sr_conv_sig0_d2 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.001 --log-results --data-log-name apr6_sr_conv_sig0_d3 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.001 --log-results --data-log-name apr6_sr_conv_sig0_d4 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.001 --log-results --data-log-name apr6_sr_conv_sig5_d0 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.001 --log-results --data-log-name apr6_sr_conv_sig5_d1 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.001 --log-results --data-log-name apr6_sr_conv_sig5_d2 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.001 --log-results --data-log-name apr6_sr_conv_sig5_d3 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.001 --log-results --data-log-name apr6_sr_conv_sig5_d4 --data digits
fi ; if [ $1 -eq 17 ] ; then
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.001 --log-results --data-log-name apr6_sr_conv_sig25_d0 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.001 --log-results --data-log-name apr6_sr_conv_sig25_d1 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.001 --log-results --data-log-name apr6_sr_conv_sig25_d2 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.001 --log-results --data-log-name apr6_sr_conv_sig25_d3 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.001 --log-results --data-log-name apr6_sr_conv_sig25_d4 --data digits
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.001 --log-results --data-log-name apr6_sr_conv_sig0_f0 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.001 --log-results --data-log-name apr6_sr_conv_sig0_f1 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.001 --log-results --data-log-name apr6_sr_conv_sig0_f2 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.001 --log-results --data-log-name apr6_sr_conv_sig0_f3 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.001 --log-results --data-log-name apr6_sr_conv_sig0_f4 --data fashion
fi ; if [ $1 -eq 18 ] ; then
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.001 --log-results --data-log-name apr6_sr_conv_sig5_f0 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.001 --log-results --data-log-name apr6_sr_conv_sig5_f1 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.001 --log-results --data-log-name apr6_sr_conv_sig5_f2 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.001 --log-results --data-log-name apr6_sr_conv_sig5_f3 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.001 --log-results --data-log-name apr6_sr_conv_sig5_f4 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.001 --log-results --data-log-name apr6_sr_conv_sig25_f0 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.001 --log-results --data-log-name apr6_sr_conv_sig25_f1 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.001 --log-results --data-log-name apr6_sr_conv_sig25_f2 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.001 --log-results --data-log-name apr6_sr_conv_sig25_f3 --data fashion
python3 synth_data_benchmark.py --sub-balanced-labels --subsample 0.001 --log-results --data-log-name apr6_sr_conv_sig25_f4 --data fashion
fi ; if [ $1 -eq 19 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig0_d0 --data digits --skip-slow-models
fi ; if [ $1 -eq 20 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig0_d1 --data digits --skip-slow-models
fi ; if [ $1 -eq 21 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig0_d2 --data digits --skip-slow-models
fi ; if [ $1 -eq 22 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig0_d3 --data digits --skip-slow-models
fi ; if [ $1 -eq 23 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig0_d4 --data digits --skip-slow-models
fi ; if [ $1 -eq 24 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig5_d0 --data digits --skip-slow-models
fi ; if [ $1 -eq 25 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig5_d1 --data digits --skip-slow-models
fi ; if [ $1 -eq 26 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig5_d2 --data digits --skip-slow-models
fi ; if [ $1 -eq 27 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig5_d3 --data digits --skip-slow-models
fi ; if [ $1 -eq 28 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig5_d4 --data digits --skip-slow-models
fi ; if [ $1 -eq 29 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig25_d0 --data digits --skip-slow-models
fi ; if [ $1 -eq 30 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig25_d1 --data digits --skip-slow-models
fi ; if [ $1 -eq 31 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig25_d2 --data digits --skip-slow-models
fi ; if [ $1 -eq 32 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig25_d3 --data digits --skip-slow-models
fi ; if [ $1 -eq 33 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig25_d4 --data digits --skip-slow-models
fi ; if [ $1 -eq 34 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig0_f0 --data fashion --skip-slow-models
fi ; if [ $1 -eq 35 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig0_f1 --data fashion --skip-slow-models
fi ; if [ $1 -eq 36 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig0_f2 --data fashion --skip-slow-models
fi ; if [ $1 -eq 37 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig0_f3 --data fashion --skip-slow-models
fi ; if [ $1 -eq 38 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig0_f4 --data fashion --skip-slow-models
fi ; if [ $1 -eq 39 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig5_f0 --data fashion --skip-slow-models
fi ; if [ $1 -eq 40 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig5_f1 --data fashion --skip-slow-models
fi ; if [ $1 -eq 41 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig5_f2 --data fashion --skip-slow-models
fi ; if [ $1 -eq 42 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig5_f3 --data fashion --skip-slow-models
fi ; if [ $1 -eq 43 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig5_f4 --data fashion --skip-slow-models
fi ; if [ $1 -eq 44 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig25_f0 --data fashion --skip-slow-models
fi ; if [ $1 -eq 45 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig25_f1 --data fashion --skip-slow-models
fi ; if [ $1 -eq 46 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig25_f2 --data fashion --skip-slow-models
fi ; if [ $1 -eq 48 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig25_f3 --data fashion --skip-slow-models
fi ; if [ $1 -eq 47 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig25_f4 --data fashion --skip-slow-models
fi ; if [ $1 -eq 49 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig0_d0 --data digits --only-slow-models
fi ; if [ $1 -eq 50 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig0_d1 --data digits --only-slow-models
fi ; if [ $1 -eq 51 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig0_d2 --data digits --only-slow-models
fi ; if [ $1 -eq 52 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig0_d3 --data digits --only-slow-models
fi ; if [ $1 -eq 53 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig0_d4 --data digits --only-slow-models
fi ; if [ $1 -eq 54 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig5_d0 --data digits --only-slow-models
fi ; if [ $1 -eq 55 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig5_d1 --data digits --only-slow-models
fi ; if [ $1 -eq 56 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig5_d2 --data digits --only-slow-models
fi ; if [ $1 -eq 57 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig5_d3 --data digits --only-slow-models
fi ; if [ $1 -eq 58 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig5_d4 --data digits --only-slow-models
fi ; if [ $1 -eq 59 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig25_d0 --data digits --only-slow-models
fi ; if [ $1 -eq 60 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig25_d1 --data digits --only-slow-models
fi ; if [ $1 -eq 61 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig25_d2 --data digits --only-slow-models
fi ; if [ $1 -eq 62 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig25_d3 --data digits --only-slow-models
fi ; if [ $1 -eq 63 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig25_d4 --data digits --only-slow-models
fi ; if [ $1 -eq 64 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig0_f0 --data fashion --only-slow-models
fi ; if [ $1 -eq 65 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig0_f1 --data fashion --only-slow-models
fi ; if [ $1 -eq 66 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig0_f2 --data fashion --only-slow-models
fi ; if [ $1 -eq 67 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig0_f3 --data fashion --only-slow-models
fi ; if [ $1 -eq 68 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig0_f4 --data fashion --only-slow-models
fi ; if [ $1 -eq 69 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig5_f0 --data fashion --only-slow-models
fi ; if [ $1 -eq 70 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig5_f1 --data fashion --only-slow-models
fi ; if [ $1 -eq 71 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig5_f2 --data fashion --only-slow-models
fi ; if [ $1 -eq 72 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig5_f3 --data fashion --only-slow-models
fi ; if [ $1 -eq 73 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig5_f4 --data fashion --only-slow-models
fi ; if [ $1 -eq 74 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig25_f0 --data fashion --only-slow-models
fi ; if [ $1 -eq 75 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig25_f1 --data fashion --only-slow-models
fi ; if [ $1 -eq 76 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig25_f2 --data fashion --only-slow-models
fi ; if [ $1 -eq 77 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig25_f3 --data fashion --only-slow-models
fi ; if [ $1 -eq 78 ] ; then
python3 synth_data_benchmark.py --log-results --data-log-name apr6_sr_conv_sig25_f4 --data fashion --only-slow-models
fi
