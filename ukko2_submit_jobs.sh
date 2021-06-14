#!/bin/bash

export LC_ALL=en_US.utf-8
export LANG=en_US.utf-8

for SEED in {0..9};
  do
    #sbatch ukko2_gpu.sh "$SEED"
    sbatch uk3.sh "SEED"
 done;
