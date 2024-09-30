#!/bin/bash
#SBATCH --job-name=Difftraining
#SBATCH --output=jeanzay_log/output.txt
#SBATCH --error=jeanzay_log/error.txt
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --account=xvy@a100
#SBATCH -C a100


module load arch/a100
module load anaconda-py3/2022.10
module load cuda/11.7.1

export LD_LIBRARY_PATH=/linkhome/rech/genuqo01/upz57sx/.conda/envs/sismic/lib
export  XLA_FLAGS="--xla_gpu_cuda_data_dir=/linkhome/rech/genuqo01/upz57sx/.conda/envs/sismic/lib"
source activate artefact
cd /gpfswork/rech/tui/upz57sx/dossier_hugo/sftp_alice
export PYTHONPATH="./"
WANDB_MODE=offline srun /gpfs/users/gabrielihu/.conda/envs/artefact/bin/python pl_training.py \
    --max_epochs 100 \
    --train_bs 8 \
    --test_bs 8 \
    --ruche \
    --no-use_feature_extractor \
    --task="Conditional_Diffusion" \
    --data_folder="control" \
    --mix_precision \
    --accumulate_grad_batches 4 \
    --lr 3e-4 \

