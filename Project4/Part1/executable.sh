#!/bin/bash
N=1000

date=$(date '+%Y-%m-%d')

# Always in bytes

for(( threads = 1; threads <= 128; threads *= 2 ))
do
    echo "-----------------------------------------------"
    echo "Running for number of threads:$threads"
    echo "-----------------------------------------------"
    ./part1-OMP $threads $N
done
