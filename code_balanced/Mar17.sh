a1=(120)
a2=(500)
a3=(1)
a4=(0.01)

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


python3 gen_balanced.py --log-name Mar17-"Test Batch-${a1[$i1]}-Train Batch-${a2[$i2]}-Epochs-${a3[$i3]}-Sampling Rate-${a4[$i4]}"  -bs ${a2[$i2]} --d-rff 10000 --rff-sigma 105 --test-batch-size ${a1[$i1]} --data digits --seed 0 --conv-gen -ks 5,5 -nc 16,8 -noise 0.0 --loss-type hermite -ep ${a3[$i3]}  --sampling-rate ${a4[$i4]} --mmd-computation mean_emb --single-release --sr-me-division 10 --sampling-multirelease

