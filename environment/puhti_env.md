# Puhti environment

In order to use comet.ml, we need to install it locally with pip.

Execute the following lines in the first time of usage. No need to do anything else in the next times.

```
module load gcc/8.3.0
module load pytorch
pip install --user comet_ml
pip install --user wandb

# wandb login $myapikey
```