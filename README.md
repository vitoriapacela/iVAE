# Binary iVAE
iVAE [VAEs and nonlinear ICA: a unifying framework](https://arxiv.org/abs/1907.04809) implementation adapted for binary data.

To set up the cluster environment, execute `environments/ukko_2_env.sh` or `environments/puhti_env.sh`.

Run the iVAE with:
`python ivae.py --obs_data_path data_5_40_1000_1.csv --mix_data_path mix_5_40_1000_1.csv --s 0 --config binary-6-2-lbfgs-100-seg.yaml --ckpt_folder='run/checkpoints/'`

Run FastICA with:
`python fastica.py --obs_data_path data_5_40_1000_1.csv --mix_data_path mix_5_40_1000_1.csv --s 0 --config binary-6-2-lbfgs-100-seg.yaml --ckpt_folder='run/checkpoints/'`

The results are stored in a CSV file in `run/` with the name of the dataset.