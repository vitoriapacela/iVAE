#!/bin/bash

# for SEED in {0..9};
for SEED in {0..99};
# for SEED in {0..4};
  do
    sbatch gpu_puhti.sh "$SEED"
 done;