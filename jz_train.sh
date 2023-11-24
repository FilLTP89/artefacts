#!/bin/bash
#SBATCH --job-name=Difftraining
#SBATCH --output=jeanzay_log/output.txt
#SBATCH --error=jeanzay_log/error.txt
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --hint=nomultithread
#SBATCH --account=xvy@a100


module load anaconda-py3/2022.10
module load cuda/11.7.1
export LD_LIBRARY_PATH=/linkhome/rech/genuqo01/upz57sx/.conda/envs/artefacts/lib
export  XLA_FLAGS="--xla_gpu_cuda_data_dir=/linkhome/rech/genuqo01/upz57sx/.conda/envs/artefacts/lib"
source activate artefacts
cd /gpfswork/rech/tui/upz57sx/dossier_hugo/artefacts
export PYTHONPATH="./"
python3 train_sam.py