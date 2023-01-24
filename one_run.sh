#!/bin/bash
#SBATCH --job-name=artefacts
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --mem=80GB
#SBATCH --output=ruche_log/output.txt
#SBATCH --error=ruche_log/error.txt
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpua100
#SBATCH --export=NONE
 

# Load the required modules
module purge
source $WORKDIR/launch_script.sh
cd $WORKDIR/artefacts
python3 train.py --gpus 3 \
--model ResUnet \
--epochs 15 \
--img_size 512 \
--learning_rate 3e-4 \
--run_name test_low_endian \
--wandb \
--saving_path model/saved_models/50_epochs \
--batch_size 8 \



