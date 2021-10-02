## Ukko2 environment setup
#!/bin/bash

# Load necessary modules
module purge
module load Python/3.6.6-foss-2018b
module load cuDNN/7.5.0.56-CUDA-10.0.130

# Create a virtual environment (in your work directory)
virtualenv /wrk/users/barimpac/research

# Activate the virtual environment
source /wrk/users/barimpac/research/bin/activate

# Install dependencies into the environment
pip --cache-dir /wrk/users/barimpac install --upgrade pip

pip --cache-dir /wrk/users/barimpac install numpy
pip --cache-dir /wrk/users/barimpac install torch torchvision
pip --cache-dir /wrk/users/barimpac install tensorflow-gpu
pip --cache-dir /wrk/users/barimpac install keras
pip --cache-dir /wrk/users/barimpac install h5py
pip --cache-dir /wrk/users/barimpac install pandas
pip --cache-dir /wrk/users/barimpac install matplotlib
pip --cache-dir /wrk/users/barimpac install setGPU
pip --cache-dir /wrk/users/barimpac install scikit-learn
pip --cache-dir /wrk/users/barimpac install scipy
pip --cache-dir /wrk/users/barimpac install jupyterlab
pip --cache-dir /wrk/users/barimpac install seaborn
pip --cache-dir /wrk/users/barimpac install glob2
pip --cache-dir /wrk/users/barimpac install PyYAML
pip --cache-dir /wrk/users/barimpac install comet-ml
pip --cache-dir /wrk/users/barimpac install wandb

/wrk/users/barimpac/research/bin/python -m pip install --upgrade pip

export LC_ALL=en_US.utf-8
export LANG=en_US.utf-8
# wandb login $myapikey