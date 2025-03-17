#!/bin/bash
#SBATCH --job-name=SDD
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=80GB
#SBATCH --output=ruche_log/singularity_output.txt
#SBATCH --error=ruche_log/singularity_error.txt
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpua100
#SBATCH --export=all
#SBATCH --exclude=ruche-gpu16,ruche-gpu13,ruche-gpu11

cd $WORKDIR
module load singularity/3.8.3/gcc-11.2.0
start_container_cmd="singularity exec --pwd /gpfs/users/gabrielihu/artefacts --bind /home/${USER}/:/home/${USER}/ --bind /gpfs/workdir/gabrielihu/artefacts:/gpfs/users/gabrielihu/artefacts --nv lightning_latest.sif"

command_to_run="python3 pl_training.py \
    --max_epochs 100 \
    --train_bs 3 \
    --test_bs 3 \
    --no-use_feature_extractor \
    --task=Conditional_Diffusion \
    --mix_precision \
    --data_folder=control \
    --no-use_deepspeed \
    --accumulate_grad_batches 4"

GAN_COMMAND="python3 pl_training.py \
    --max_epochs 100 \
    --train_bs 8 \
    --test_bs 8 \
    --task=GAN \
    --ruche \
    --no-use_deepspeed \
    --data_folder=complete"

VGG_COMMAND="python3 pl_training.py \
    --max_epochs 100 \
    --train_bs 64 \
    --test_bs 64 \
    --no-use_deepspeed \
    --no-use_feature_extractor \
    --task=Classification \
    --data_folder=complete "



srun $start_container_cmd /bin/bash -c "$GAN_COMMAND"