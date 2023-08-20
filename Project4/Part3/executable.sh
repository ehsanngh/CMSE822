#!/bin/bash
N=1000

date=$(date '+%Y-%m-%d')

# Always in bytes
for(( size = 1; size <=4; size *= 2))
do
    for(( threads = 1; threads <= 32; threads *= 2 ))
    do
        echo "-----------------------------------------------"
        echo "Running for number of threads:$threads and nodes:$size"
        echo "-----------------------------------------------"
        mpirun -n $size ./part3-OMP $threads $N
    done
done
