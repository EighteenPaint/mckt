#for  model_type in staktc staktns staktsk stakt;
for  th in 100 1000 10000;
do
  {
        python -u main.py  --dataset assist2017_pid --m 2 --n 2 --kernel_size 2  --th ${th} >> th${th}_assist2017_pid.log 2>&1 &&
        wait
  }
done
