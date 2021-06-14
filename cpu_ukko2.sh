#!/bin/bash
#SBATCH -M ukko2
#SBATCH -p short
#SBATCH --job-name=iVAE
#SBATCH -o result.txt
#SBATCH --workdir=/proj/barimpac/beyond-NonSENS/iVAE
#SBATCH -c 1
#SBATCH -t 10:00
#SBATCH --mem=10M

module purge                             # Purge modules for a clean start
module load Python/3.6.6-foss-2018b
module load cuDNN/7.5.0.56-CUDA-10.0.130

source /wrk/users/barimpac/research/bin/activate

# Executable here
python3 main.py --config discrete-full-ivae.yaml --n-sims 1 --m 1 --ckpt_folder='run/checkpoints/'
