#!/bin/bash

# sbatch will read any lines that start with "#SBATCH" and will
# add what follows as if they were command line parameters.

#SBATCH --job-name=ivae
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=100Gb

# Load the necessary modules
module load miniconda/3 pytorch/1.8.1

# Activate a virtual environment, if needed:
conda activate research

# Run the script
python main.py --config binary-6-2-fast_ica.yaml --n-sims 3 --m 2.0 --ckpt_folder='run/checkpoints/' --s "$@" > "$1"_b_6_2.log
