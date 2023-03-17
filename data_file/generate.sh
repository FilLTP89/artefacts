#!/bin/bash
#SBATCH --job-name=artefacts
#SBATCH --nodes=1
#SBATCH --output=../ruche_log/output_process.txt
#SBATCH --error=../ruche_log/error_process.txt
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=20
#SBATCH --partition=cpu_short

module purge
source $WORKDIR/launch_script.sh
python3 processing_dicom.py