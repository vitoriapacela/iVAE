# Binary iVAE
Identifiable VAE (iVAE) implementation adapted for binary data, based on [Ilyes Khemakhem's code](https://github.com/ilkhem/icebeem/tree/master/models/ivae).

Install the virtual environment and dependencies listed at `environment/` (see `conda_env.sh` for example).


To reproduce some of the baseline results from the paper [Binary independent component analysis: a non-stationarity-based approach](https://proceedings.mlr.press/v180/hyttinen22a.html), run:

Run the iVAE with:
`python ivae.py --obs_data_path data_5_40_1000_1.csv --mix_data_path mix_5_40_1000_1.csv --s 0 --config binary-6-2-lbfgs-100-seg.yaml --ckpt_folder='run/checkpoints/'`

Run FastICA with:
`python fastica.py --obs_data_path data_5_40_1000_1.csv --mix_data_path mix_5_40_1000_1.csv --s 0 --config binary-6-2-lbfgs-100-seg.yaml --ckpt_folder='run/checkpoints/'`

The results are stored in a CSV file in `run/` with the name of the dataset.
