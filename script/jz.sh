#!/bin/bash
#SBATCH --job-name=Generate
#SBATCH --output=jeanzay_log/generate_output.txt
#SBATCH --error=jeanzay_log/generate_error.txt
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --hint=nomultithread
#SBATCH --account=xvy@a100

module load anaconda-py3/2023.09
module load cuda/11.8.0
module load gcc/11.3.0



WANDB__SERVICE_WAIT=1000
export LD_LIBRARY_PATH=/linkhome/rech/tui/upz57sx/.conda/envs/sismic/lib
export  XLA_FLAGS="--xla_gpu_cuda_data_dir=/linkhome/rech/tui/upz57sx/.conda/envs/sismic/lib"
source activate sismic
cd /gpfswork/rech/tui/upz57sx/dossier_hugo/sftp_alice
export PYTHONPATH="./"
python3 generate/generate_small_dance_diffusion.py



