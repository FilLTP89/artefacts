#!/bin/bash
#SBATCH --job-name=artefacts
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --mem=80GB
#SBATCH --output=ruche_log/output.txt
#SBATCH --error=ruche_log/error.txt
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpua100
#SBATCH --export=NONE


module purge
source $WORKDIR/launch_script.sh
#cd $WORKDIR/artefacts
#python3 train.py --model VGG19 --epochs 20 --batch_size 32 --wandb --saving_path "model/saved_models/VGG19/" --dicom --learning_rate 1.5e-4 --save_weights
#python3 train.py --model MedGAN --epochs 50 --batch_size 3 --wandb --saving_path "model/saved_models/MedGAN/" --dicom --learning_rate 4e-6 --save_weights 
#python3 train.py --model ResUnet --epochs 20 --batch_size 8 --wandb --saving_path "model/saved_models/ResUnet/" --big_endian
python3 train.py --model MedGAN --epochs 5 --batch_size 1 --dicom  --one_batch_training --learning_rate 2e-6 --wandb --saving_path "model/saved_models/MedGAN/" --save_weights
#python3 train.py --model DeepMAR --epochs 5 --batch_size 1 --dicom 
