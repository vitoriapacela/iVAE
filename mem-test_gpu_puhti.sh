#!/bin/bash
#SBATCH --job-name=iVAE
#SBATCH --account=Project_2002842
#SBATCH --partition=gputest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=300G

module purge
# module load gcc/8.3.0
module load pytorch/1.4

srun python3 main.py --config real-full-ivae-u-simple-prior-logv-10.yaml --n-sims 1 --m 0.5 --s 0
