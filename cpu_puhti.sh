#!/bin/bash
#SBATCH --job-name=iVAE
#SBATCH --account=Project_2002842
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8000
#SBATCH --time=3-00:00:00

module purge
# module load gcc/8.3.0
module load pytorch/1.4

# srun python3 main.py --config binary-3-2-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_3_2.log
# srun python3 main.py --config binary-4-2-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_4_2.log
# srun python3 main.py --config binary-5-2-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_5_2.log
# srun python3 main.py --config binary-5-3-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_5_3.log
# srun python3 main.py --config binary-6-2-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_6_2.log
# srun python3 main.py --config binary-6-3-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_6_3.log
# srun python3 main.py --config binary-7-2-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_7_2.log
# srun python3 main.py --config binary-7-3-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_7_3.log
# srun python3 main.py --config binary-8-2-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_8_2.log
# srun python3 main.py --config binary-8-3-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_8_3.log
# srun python3 main.py --config binary-9-2-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_9_2.log
# srun python3 main.py --config binary-9-3-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_9_3.log
# srun python3 main.py --config binary-10-2-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_10_2.log
# srun python3 main.py --config binary-10-3-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_10_3.log
# srun python3 main.py --config binary-10-4-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_10_4.log
# srun python3 main.py --config binary-6-2-lbfgs-1000.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_6_2_1000nps.log
# srun python3 main.py --config binary-6-2-lbfgs-500.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_6_2_500nps.log
# srun python3 main.py --config binary-6-2-lbfgs-100.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_6_2_100nps.log
# srun python3 main.py --config binary-6-2-lbfgs-4000.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_6_2_4000nps.log
# srun python3 main.py --config binary-6-2-lbfgs-50.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_6_2_50nps.log
# srun python3 main.py --config binary-6-2-lbfgs-30-seg.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_6_2_30seg.log
# srun python3 main.py --config binary-6-2-lbfgs-20-seg.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_6_2_20seg.log
# srun python3 main.py --config binary-6-2-lbfgs-10-seg.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_6_2_10seg.log
# srun python3 main.py --config binary-6-2-lbfgs-100-seg.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_6_2_100seg.log
# srun python3 main.py --config binary-6-2-lbfgs-1-seg.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_6_2_1seg.log
# srun python3 main.py --config binary-100-10-lbfgs.yaml --n-sims 3 --m 2.0 --s 0
# srun python3 main.py --config binary-100-10-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_100_10.log
# srun python3 main.py --config binary-9-4-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_9_4.log
# srun python3 main.py --config binary-2-2-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_2.log
# srun python3 main.py --config binary-3-3-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_3.log
# srun python3 main.py --config binary-4-4-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_4.log
# srun python3 main.py --config binary-5-5-lbfgs.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_5.log
# srun python3 main.py --config binary-6-2-lbfgs-50.yaml --n-sims 3 --m 2.0 --s 86
# srun python3 main.py --config binary-6-2-lbfgs-100-seg.yaml --n-sims 3 --m 2.0 --s 26
#srun python3 main.py --config binary-6-2-lbfgs-100s.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_6_2_100s.log
# srun python3 main.py --config binary-6-2-lbfgs-20s.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_6_2_20s.log
# srun python3 main.py --config binary-6-2-lbfgs-10s.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_6_2_10s.log
# srun python3 main.py --config binary-6-2-lbfgs-1s.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_6_2_1s.log
# srun python3 main.py --config binary-6-2-fast_ica.yaml --n-sims 3 --m 2.0 --s "$@" > "$1"_b_6_2.log
srun python3 main.py --config binary-6-2-fast_ica.yaml --n-sims 3 --m 2.0 --s 0
