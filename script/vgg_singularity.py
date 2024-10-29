#!/bin/bash
#SBATCH --job-name=artefact
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB
#SBATCH --output=ruche_log/vgg_output.txt
#SBATCH --error=ruche_log/vgg_error.txt
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpua100
#SBATCH --export=NONE
#SBATCH --exclude=ruche-gpu16,ruche-gpu13,ruche-gpu11
