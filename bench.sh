#!/bin/bash

export KMP_AFFINITY=granularity=fine,compact,1,0

make clean && make

for b in ref-asm.bin ref.bin dev.bin dev-asm.bin;
do
  for t in {1..56..1};
  do
    export OMP_NUM_THREADS=$t
    ./${b} 1 1120000000 2>&1 | tee -a ${b}.log
  done
done
