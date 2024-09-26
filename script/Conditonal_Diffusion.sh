#!/bin/bash
#SBATCH --job-name=artefact
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB
#SBATCH --output=ruche_log/diffusion_output.txt
#SBATCH --error=ruche_log/diffusion_error.txt
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpua100
#SBATCH --export=NONE
#SBATCH --exclude=ruche-gpu16,ruche-gpu13,ruche-gpu11

module load anaconda3/2022.10/gcc-11.2.0 
module load gcc/11.2.0/gcc-4.8.5
module load cuda/11.8.0/gcc-11.2.0
module load strace/5.2/gcc-9.2.0

export NCCL_P2P_DISABLE=1
export LD_LIBRARY_PATH=/gpfs/users/gabrielihu/.conda/envs/artefact/lib
export  XLA_FLAGS="--xla_gpu_cuda_data_dir=/gpfs/users/gabrielihu/.conda/envs/artefact/lib/"
source activate artefact
cd $WORKDIR/artefacts/
export PYTHONPATH="./"
export WANDB__SERVICE_WAIT=1000

WANDB_MODE=offline python3 pl_training.py \
    --max_epochs 100 \
    --train_bs 2 \
    --test_bs 2 \
    --ruche \
    --no-use_feature_extractor \
    --task="Conditional_Diffusion" \
    --mix_precision \
    --data_folder="control" \
    --one_batch \





