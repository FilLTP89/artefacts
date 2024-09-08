#!/bin/bash
#SBATCH --job-name=artefact
#SBATCH --output=jeanzay_log/output.txt
#SBATCH --error=jeanzay_log/error.txt
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --hint=nomultithread
#SBATCH --account=xvy@a100

module load anaconda-py3/2023.09
module load cuda/11.8.0

export NCCL_P2P_DISABLE=1
export LD_LIBRARY_PATH=/linkhome/rech/tui/upz57sx/.conda/envs/sismic/lib
export  XLA_FLAGS="--xla_gpu_cuda_data_dir=/linkhome/rech/genuqo01/upz57sx/.conda/envs/artefact/lib"
source activate artefact
chmod +x /lustre/fswork/projects/rech/xvy/upz57sx/artefact/pl_training.py

cd /lustre/fswork/projects/rech/xvy/upz57sx/artefact
export PYTHONPATH="./"
WANDB_MODE=offline srun python3 /lustre/fswork/projects/rech/xvy/upz57sx/artefact/pl_training.py \
    --max_epochs 100 \
    --train_bs 16 \
    --test_bs 16 \


