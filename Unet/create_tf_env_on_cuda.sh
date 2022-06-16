#!/bin/bash
source load_conda_cuda.sh
conda create --name tf
source activate tf
conda install -c conda-forge cudatoolkit cudnn
pip install tensorflow
