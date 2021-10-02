#!/bin/bash
#SBATCH --job-name=test
#SBATCH --workdir=/wrk/barimpac/beyond-NonSENS
#SBATCH -o /wrk/barimpac/beyond-NonSENS/result.txt
#SBATCH -p gpu
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH --mem=10M
#SBATCH -t 7-0  # running time
#SBATCH --mem-per-cpu=100G

module purge
module load Python/3.6.6-foss-2018b
module load cuDNN/7.5.0.56-CUDA-10.0.130

source /wrk/users/barimpac/research/bin/activate

# Executable here
#python /wrk/$USER/beyond-NonSENS/test.py
jupyter lab --no-browser --port=8889
