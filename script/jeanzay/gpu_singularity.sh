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
#SBATCH --hint=nomultithread        


module load singularity/3.8.5

# Command to run inside the container
command_to_run_GAN="python3 pl_training.py \
    --max_epochs 100 \
    --train_bs 2 \
    --test_bs 2 \
    --task=GAN \
    --mix_precision \
    --no-use_feature_extractor \
    --data_folder=control \
    --accumulate_grad_batches 1"

# Singularity execution with GPU support
start_container_cmd="singularity exec --pwd /lustre/fswork/projects/rech/xvy/ucn85lb/artefacts \
        -B /lustre/fswork/projects/rech/xvy/ucn85lb:/lustre/fswork/projects/rech/xvy/ucn85lb/ \
         --bind /lustre/fswork/projects/rech/xvy/ucn85lb/artefacts:/lustre/fswork/projects/rech/xvy/ucn85lb/artefacts \
         --nv /lustre/fsn1/singularity/images/ucn85lb/lightning_latest.sif"

# Run the command inside Singularity
srun $start_container_cmd $command_to_run_GAN