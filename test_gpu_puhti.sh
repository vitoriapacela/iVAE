#!/bin/bash
#SBATCH --job-name=iVAE
#SBATCH --account=Project_2002842
#SBATCH --partition=gputest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8000
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:v100:1

module purge
# module load gcc/8.3.0
module load gcc/8.3.0 cuda/10.1.168
module load pytorch/1.4

# srun python3 main.py --config binary-2-1-2-seg-gd.yaml --n-sims 1 --m 2.0 --s 0
# srun python3 main.py --config binary-2-1-1-seg-gd.yaml --n-sims 1 --m 2.0 --s 0
# srun python3 main.py --config binary-3-2-1-seg-gd.yaml --n-sims 1 --m 2.0 --s 0
# srun python3 main.py --config binary-4-3-gd.yaml --n-sims 1 --m 2.0 --s 0
# srun python3 main.py --config binary-10-3-var1-gd.yaml --n-sims 1 --m 2.0 --s 0
# srun python3 main.py --config binary-10-3-mean0-gd.yaml --n-sims 1 --m 0.0 --s 0
# srun python3 main.py --config binary-3-2-lbfgs.yaml --n-sims 1 --m 2.0 --s 0
# srun python3 main.py --config binary-3-2-lbfgs.yaml --n-sims 2 --m 2.0 --s 0
srun python3 main.py --config binary-6-2-lbfgs-10.yaml --n-sims 1 --m 2.0 --s 0