#!/bin/bash
#SBATCH --job-name=Difftraining
#SBATCH --output=jeanzay_log/output.txt
#SBATCH --error=jeanzay_log/error.txt
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --account=xvy@a100
#SBATCH -C a100


export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker0
cd /lustre/fswork/projects/rech/xvy/ucn85lb/artefacts/
module load singularity/3.8.5

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

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
    --task=Classification \
    --data_folder=complete "

start_container_cmd="singularity exec --pwd /lustre/fswork/projects/rech/xvy/ucn85lb/medical_project/artefacts/ -B /lustre/fswork/projects/rech/xvy/ucn85lb:/lustre/fswork/projects/rech/xvy/ucn85lb/ --bind /lustre/fswork/projects/rech/xvy/ucn85lb/medical_project/artefacts:/lustre/fswork/projects/rech/xvy/ucn85lb/artefacts --nv /lustre/fsn1/singularity/images/ucn85lb/artefact.sif"

srun $start_container_cmd /bin/bash -c "$GAN_command"
