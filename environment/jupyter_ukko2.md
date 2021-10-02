### In Ukko2:
You may need to run the following line in the first time.

`rm -r ~/.ipython/ ~/.jupyter/ ~/.local`

Otherwise, jump straight here. The following line submits an interactive job to a gpu for 4 hours.

`interactive gpu 1 1`

That will submit an interactive job to a GPU. If preferable, `submit_gpu.sh` can be used instead. Then, load the necessary modules and activate the environment before launching jupyter.

```
module purge
module load Python/3.6.6-foss-2018b
module load cuDNN/7.5.0.56-CUDA-10.0.130

source /wrk/users/barimpac/research/bin/activate

jupyter lab --no-browser --port=8889 --notebook-dir=$WRKDIR
```

### In your local computer:
`ssh -N -f -L localhost:8889:localhost:8889 ukko2.cs.helsinki.fi`

Notice that if you are not in the university network, the host will not be reachable and additional settings are needed.
