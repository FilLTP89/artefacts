#!/bin/bash
source load_conda_cuda.sh
source activate tf
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfs/users/gattif/.conda/envs/tf/lib

