#!/bin/bash
#SBATCH -J test
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=120GB
#SBATCH --time=24:00:00
#SBATCH --partition=gpua100
#SBATCH --output=oinfo.txt
#SBATCH --error=hinfo.txt 
#SBATCH --mail-type=FAIL
#SBATCH --export=NONE 

source load_tf.sh
python3 Unet_artefacts.py --trDatabase ${WORKDIR}/train_artefacts --tsDatabase ${WORKDIR}/test_artefacts --vdDatabase ${WORKDIR}/test_artefacts --epochs $1
