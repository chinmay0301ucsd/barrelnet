# barrels (put paper name here or something)

You want barrels? We got barrels.

## Environment setup

Just run this, I'll deal with a requirements file later. If anything is still missing after
installation just pip install it.

```shell
git clone https://github.com/chinmay0301ucsd/barrelnet.git
cd barrelnet
git submodule update --init --recursive
conda create --name barrels python=3.11
conda activate barrels
conda install -c nvidia cuda
export CUDA_HOME=$CONDA_PREFIX
pip install git+https://github.com/jerukan/lang-segment-anything.git
pip install git+https://github.com/google-research/visu3d.git
pip install jupyter plotly dill pyransac3d open3d transforms3d roma pyrender mitsuba
```
