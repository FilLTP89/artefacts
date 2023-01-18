#!/bin/bash
#SBATCH --job-name=artefacts
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=80GB
#SBATCH --output=$WORKDIR/artefacts/ruche_log/output.txt
#SBATCH --error=$WORKDIR/artefacts/ruche_log/error.txt
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpua100
#SBATCH --export=NONE

# Load the required modules

module purge
source $WORKDIR/launch_script.sh
cd $WORKDIR/artefacts
wandb sweep sweep.yaml
wandb agent
