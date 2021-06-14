#!/bin/bash
#SBATCH --job-name=iVAE
#SBATCH --account=Project_2002842
#SBATCH --partition=test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=14000
#SBATCH --time=15:00

module purge
# module load gcc/8.3.0
module load pytorch/1.4

# srun python3 main.py --config binary-5-adam.yaml --n-sims 3 --m 2.0 --s 3
# srun python3 main.py --config binary-5-gd.yaml --n-sims 3 --m 2.0 --s 3
# srun python3 main.py --config binary-5-gd-Copy8.yaml --n-sims 3 --m 2.0 --fix_prior_mean --s 33
# srun python3 main.py --config cont-5-gd.yaml --n-sims 3 --m 2.0 --s 33
# srun python3 main.py --config binary-3-2-lbfgs.yaml --n-sims 1 --m 2.0 --s 0
# srun python3 main.py --config binary-6-2-lbfgs-10.yaml --n-sims 1 --m 2.0 --s 0
# srun python3 main.py --config binary-6-2-lbfgs-50.yaml --n-sims 1 --m 2.0 --s 0
srun python3 main.py --config binary-6-2-lbfgs-1-seg.yaml --n-sims 1 --m 2.0 --s 0