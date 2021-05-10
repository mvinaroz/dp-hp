#!/bin/bash
a1=(200)

a2=(0 1 2 3 4)

a3=(10)

a4=(5 10 20 40 50 80 100)

a5=(0.15)


let "i1=$1%${#a1[@]}"

let "p1=$1/${#a1[@]}"

let "i2=$p1%${#a2[@]}"

let "p2=$p1/${#a2[@]}"

let "i3=$p2%${#a3[@]}"

let "p3=$p2/${#a3[@]}"

let "i4=$p3%${#a4[@]}"

let "p4=$p3/${#a4[@]}"

let "i5=$p4%${#a5[@]}"

let "p5=$p4/${#a5[@]}"


echo "${a1[$i1]}"

echo "${a2[$i2]}"

echo "${a3[$i3]}"

echo "${a4[$i4]}"

echo "${a5[$i5]}"


python3 Me_sum_kernel_args.py --log-name fashion_priv_"${a5[$i5]}-bs-${a1[$i1]}-seed-${a2[$i2]}-epochs-${a3[$i3]}-hp-${a4[$i4]}" --data fashion --model_name CNN -bs  ${a1[$i1]} -lr 0.01 --seed ${a2[$i2]} -ep ${a3[$i3]} --order-hermite ${a4[$i4]} --kernel-length ${a5[$i5]} --is-private True 


