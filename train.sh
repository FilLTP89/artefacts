#!/bin/bash
#SBATCH --job-name=artefacts
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --mem=80GB
#SBATCH --output=ruche_log/output.txt
#SBATCH --error=ruche_log/error.txt
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpua100
#SBATCH --export=NONE


module purge
source $WORKDIR/launch_script.sh
cd $WORKDIR/artefacts
#python3 train.py --model VGG19 --epochs 2 --batch_size 32 --wandb --saving_path "model/saved_models/VGG19/"
python3 train.py --model MedGAN --epochs 3 --batch_size 3 --wandb --saving_path "model/saved_models/MedGAN/" --big_endian --learning_rate 0.0002
#python3 train.py --model MedGAN --epochs 3 --batch_size 3 --wandb --saving_path "model/saved_models/MedGAN/" 