## Virtual environment for CPU
# It is recommended to first create and activate the environment.
# And then install all the packages using this script.
conda create --name research python=3.7 -y
conda activate research

conda install numpy -y
conda install pandas -y
conda install scikit-learn -y
conda install -c conda-forge matplotlib -y
conda install -c anaconda scipy -y
conda install -c conda-forge jupyterlab -y
conda install -c anaconda seaborn -y
conda install -c conda-forge glob2 -y
conda install pytorch torchvision -c pytorch -y
conda install 'tensorflow=*=mkl*' -y
conda install -c conda-forge keras -y
conda install -c conda-forge jax -y
pip install pytorch-lightning -y
conda install -c conda-forge ipywidgets -y
conda install -c anaconda pyyaml -y
conda install -c comet_ml -c conda-forge comet_ml -y
conda install -c conda-forge graphql-core=2.0 -y
conda install -c conda-forge wandb -y
conda install -c r rpy2

conda update --all -y
conda clean --all -y

# wandb login $myapikey