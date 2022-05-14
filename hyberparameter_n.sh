#for  model_type in staktc staktns staktsk stakt;
for dataset in assist2017_pid assist2009_pid assist2015
do
{
  for n in 1 2 3 4;
    do
      {
            python -u main.py  --dataset ${dataset} --m 2 --n ${n} --kernel_size 2  >> param_${n}_${dataset}.log 2>&1 &&
            wait
      }
    done
}
done