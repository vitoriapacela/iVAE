#!/bin/bash

# for SEED in {0..9};
# for SEED in {0..4};
# for SEED in {5..9};
# for SEED in {10..19};
# for SEED in {0..19};
# for SEED in {20..99};
for SEED in {0..99};
# for SEED in {10..99};
# for SEED in {69..99};
  do
    sbatch cpu_puhti.sh "$SEED"
 done;
