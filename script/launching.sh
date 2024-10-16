module load anaconda3/2022.10/gcc-11.2.0 
module load gcc/11.2.0/gcc-4.8.5
module load cuda/11.8.0/gcc-11.2.0
source activate artefact
export PYTHONPATH="./"
export WANDB__SERVICE_WAIT=1000
