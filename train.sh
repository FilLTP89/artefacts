#!/bin/bash
#SBATCH --job-name=artefacts
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=80GB
#SBATCH --output=ruche_log/output.txt
#SBATCH --error=ruche_log/error.txt
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpua100
#SBATCH --export=NONE


module purge
source $WORKDIR/launch_script.sh
cd $WORKDIR/artefacts
python3 train_vgg19.py 
#python3 train.py --model MedGAN --epochs 3 --batch_size 3 --wandb --saving_path "model/saved_models/MedGAN/"