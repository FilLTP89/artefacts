#!/bin/bash
#SBATCH --job-name=artefacts
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=120GB
#SBATCH --output=./ruche_log/output.txt
#SBATCH --error=./ruche_log/error.txt
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpu
#SBATCH --export=NONE
 

# Load the required modules
module purge
source $WORKDIR/launch_script.sh
python3 train.py --gpus 1 \
--model ResUnet \
--epochs 50 \
--img_size 512 \
--run_name ResUnet_50epochs
--wandb \
--save \

