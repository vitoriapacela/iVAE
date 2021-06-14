#!/bin/bash
#SBATCH --job-name=iVAE
#SBATCH --account=Project_2002842
#SBATCH --partition=gputest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=10000
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:v100:1

module load pytorch/1.4
srun python3 discrete.py --method ica --seed 3
