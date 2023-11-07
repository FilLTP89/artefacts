#!/bin/bash
#SBATCH --job-name=artefacts
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB
#SBATCH --output=log/output.txt
#SBATCH --error=og/error.txt
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpua100
#SBATCH --export=NONE


module purge
source $WORKDIR/launch_script.sh
cd $WORKDIR/artefacts
export PYTHONPATH="./"
python3 test_amanda.py --generate=$1 --dicom=$2 --low=$3 --acquisition_number=$4