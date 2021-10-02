# Environment

`conda_env.sh` sets up a conda environment (called `research`) in your local computer. It requires Miniconda installed.

`ukko2.sh` sets up a pip virtual environment (called `research`) in Ukko2.

`gpu_ukko2.sh` is an example script to submit a job to a GPU in Ukko2.

`jupyter_ukko2.md` has instructions to launch jupyter lab notebooks in Ukko2.

`tensorboard.md` has instructions to launch Tensorboard in both Puhti and Ukko2.

To launch jupyter notebooks in Puthi:
`module load pytorch`
`sinteractive --account beyond-NonSENS start-jupyter-server -m 5000`