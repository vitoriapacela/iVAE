#!/bin/bash
#SBATCH --job-name=iVAE
#SBATCH --account=Project_2002842
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem-per-cpu=8000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

module purge
# module load gcc/8.3.0
module load pytorch/1.4

# srun python3 main.py --config real-full-ivae-u-simple-prior-logv-5-Copy1.yaml --n-sims 3 --m 0.5 --s "$@" > "$1"_c2.log
srun python3 main.py --config binary-5-adam-Copy7.yaml --n-sims 3 --m 2.0 --s 33 --fix_prior_mean