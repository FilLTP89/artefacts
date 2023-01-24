#!/bin/bash
#SBATCH --job-name=artefacts
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --mem=80GB
#SBATCH --output=ruche_log/output2.txt
#SBATCH --error=ruche_log/error2.txt
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpua100
#SBATCH --export=NONE
 

# Load the required modules
module purge
source $WORKDIR/launch_script.sh
cd $WORKDIR/artefacts
python3 run_sweep2.py --big_endian

