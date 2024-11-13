#!/bin/bash
#SBATCH --job-name=artefact
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --mem=40GB
#SBATCH --output=ruche_log/output.txt
#SBATCH --error=ruche_log/error.txt
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpua100
#SBATCH --export=NONE
#SBATCH --exclude=ruche-gpu16,ruche-gpu13,ruche-gpu11

module load singularity/3.8.3/gcc-11.2.0
cd $WORKDIR
start_container_cmd="singularity exec --pwd /gpfs/workdir/gabrielihu -B /gpfs/workdir:/gpfs/workdir --bind /home/${USER}/:/home/${USER}/ --bind /gpfs/workdir/gabrielihu/artefacts:/gpfs/users/gabrielihu/artefacts --nv lightning_latest.sif"


command_to_run="WANDB_MODE=offline python3 pl_training.py \
    --max_epochs 100 \
    --train_bs 3 \
    --test_bs 3 \
    --no-use_feature_extractor \
    --task=Conditional_Diffusion \
    --mix_precision \
    --data_folder=control \
    --no-use_deepspeed \
    --accumulate_grad_batches 4"

GAN_command="WANDB_MODE=offline python3 pl_training.py \
    --max_epochs 100 \
    --train_bs 16 \
    --test_bs 16 \
    --no-use_feature_extractor \
    --task=GAN \
    --no-use_deepspeed \
    --data_folder=complete"

VGG_COMMAND="WANDB_MODE=offline python3 pl_training.py \
    --max_epochs 100 \
    --train_bs 64 \
    --test_bs 64 \
    --no-use_deepspeed \
    --no-use_feature_extractor \
    --task=Classification \
    --data_folder=complete "



srun $start_container_cmd /bin/bash -c "$VGG_COMMAND"