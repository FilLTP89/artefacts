#!/bin/bash
#SBATCH --job-name=artefact_matlab
#SBATCH --output=%x.o%j 
#SBATCH --time=00:20:00 
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpua10
#SBATCH --output=output.txt
#SBATCH --error=error.txt


# Module load
module purge
module load cuda/11.7.0/gcc-11.2.0
module load gcc/8.4.0/gcc-4.8.5
module load matlab/R2020a/intel-19.0.3.199
matlab -nodisplay -r read_images
matlab -nodisplay -r reconstruct_volume