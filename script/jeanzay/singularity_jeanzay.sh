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
#SBATCH --hint=nomultithread
#SBATCH --account=xvy@a100
#SBATCH -C a100


cd /lustre/fswork/projects/rech/xvy/ucn85lb/artefacts/


command_to_run="WANDB_MODE=offline python3 pl_training.py \
    --max_epochs 100 \
    --train_bs 3 \
    --test_bs 3 \
    --ruche \
    --no-use_feature_extractor \
    --task=Conditional_Diffusion \
    --mix_precision \
    --data_folder=control \
    --accumulate_grad_batches 4"

start_container_cmd="singularity exec --pwd /lustre/fswork/projects/rech/xvy/ucn85lb/artefacts/ -B /lustre/fswork/projects/rech/xvy/ucn85lb:/lustre/fswork/projects/rech/xvy/ucn85lb/ --bind /lustre/fswork/projects/rech/xvy/ucn85lb/artefacts:/lustre/fswork/projects/rech/xvy/ucn85lb/artefacts --nv /lustre/fsn1/singularity/images/ucn85lb/lightning_latest.sif"

srun --ntasks-per-node=2 --gres=gpu:2 $start_container_cmd /bin/bash -c "$command_to_run"
