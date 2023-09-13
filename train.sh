#!/bin/bash
#SBATCH --job-name=artefacts
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --mem=80GB
#SBATCH --output=log/output.txt
#SBATCH --error=log/error.txt
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpua100
#SBATCH --export=NONE


module purge
source $WORKDIR/launch_script.sh
#cd $WORKDIR/artefacts


python3 train.py --model MedGAN --epochs 200 --batch_size 6 --wandb --saving_path "model/saved_models/MedGAN/" --big_endian --no-dicom --learning_rate 1e-6 
#python3 train.py --model VGG19 --epochs 20 --batch_size 32 --wandb --saving_path "model/saved_models/VGG19/" --dicom --learning_rate 1.5e-4 --save_weights
#python3 train.py --model MedGAN --epochs 200 --batch_size 6 --wandb --saving_path "model/saved_models/MedGAN/" --big_endian --no-dicom --learning_rate 4e-6 --save_weights 
#python3 train.py --model ResUnet --epochs 20 --batch_size 8 --wandb --saving_path "model/saved_models/ResUnet/" --big_endian
#python3 train.py --model MedGAN --epochs 5 --batch_size 1 --dicom  --one_batch_training --learning_rate 2e-6 --saving_path "model/saved_models/MedGAN/" --save_weights
#python3 train.py --model DeepMAR --epochs 5 --batch_size 1 --dicom 

#python3 train.py --model smResunet --epochs 200 --batch_size 8 --wandb --saving_path "model/saved_models/ResUnet/" --segmentation --learning_rate 3e-4