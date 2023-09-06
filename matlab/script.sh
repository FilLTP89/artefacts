#!/bin/bash
#SBATCH --job-name=artefact_matlab
#SBATCH --output=%x.o%j 
#SBATCH --time=00:20:00 
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpua100
#SBATCH --output=output.txt
#SBATCH --error=error.txt


# Module load
module purge
module load matlab/R2020a/intel-19.0.3.199
matlab -nodisplay -r read_images