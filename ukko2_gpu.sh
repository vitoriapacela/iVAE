#!/bin/bash
#SBATCH -M ukko2
#SBATCH -p gpu-short
#SBATCH --job-name=iVAE
#SBATCH --workdir=/proj/barimpac/beyond-NonSENS/iVAE/
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -t 1-00:00:00  # running time
#SBATCH --mem=10G

module purge
module load Python/3.6.6-foss-2018b
module load cuDNN/7.5.0.56-CUDA-10.0.130

source /wrk/users/barimpac/research/bin/activate

# Executable here
srun python3 main.py --config binary-2-1-gd.yaml --n-sims 3 --m 2.0 --s 1 --ckpt_folder='run/checkpoints/'
# srun python3 main.py --config binary-3-2-gd.yaml --n-sims 3 --m 2.0 --s 1 --ckpt_folder='run/checkpoints/'