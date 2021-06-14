#!/bin/bash

module purge
module load Python/3.6.6-foss-2018b
module load cuDNN/7.5.0.56-CUDA-10.0.130

source /wrk/users/barimpac/research/bin/activate

export LC_ALL=en_US.utf-8
export LANG=en_US.utf-8

tensorboard --logdir=tensorboard/ivae/
