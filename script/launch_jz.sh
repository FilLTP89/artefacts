#!/bin/bash


module load anaconda-py3/2023.09
module load cuda/11.8.0

export NCCL_P2P_DISABLE=1
export LD_LIBRARY_PATH=/linkhome/rech/tui/upz57sx/.conda/envs/sismic/lib
export  XLA_FLAGS="--xla_gpu_cuda_data_dir=/linkhome/rech/genuqo01/upz57sx/.conda/envs/artefact/lib"
conda activate artefact

chmod +x /lustre/fswork/projects/rech/xvy/upz57sx/artefact/pl_training.py

cd /lustre/fswork/projects/rech/xvy/upz57sx/artefact
export PYTHONPATH="./"

