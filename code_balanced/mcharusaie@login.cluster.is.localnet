a1=(60 120 300 600 1200)
a2=(100 200 500 1000 2000 5000)
a3=(1 2 3 4 5 6 98 10)
a4=(0.01 0.02 0.04 0.07 0.11 0.15)

let "i1=$1%${#a1[@]}"
let "p1=$1/${#a1[@]}"
let "i2=$p1%${#a2[@]}"
let "p2=$p1/${#a2[@]}"
let "i3=$p2%${#a3[@]}"
let "p3=$p2/${#a3[@]}"
let "i4=$p3%${#a4[@]}"
let "p4=$p3/${#a4[@]}"

echo "${a1[$i1]}"
echo "${a2[$i2]}"
echo "${a3[$i3]}"
echo "${a4[$i4]}"

python3 gen_balanced.py --log-name Feb22-"Test Batch-${a1[$i1]}-Train Batch-${a2[$i2]}-Epochs-${a3[$i3]}-Sampling Rate-${a4[$i4]}"  -bs ${a2[$i2]} --d-rff 10000 --rff-sigma 105 --test-batch-size ${a1[$i1]} --data digits --seed 0 --conv-gen -ks 5,5 -nc 16,8 -noise 0.0 --loss-type hermite -ep ${a3[$i3]}  --sampling-rate ${a4[$i4]}

