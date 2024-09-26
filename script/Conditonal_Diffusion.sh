
source activate artefact
cd $WORKDIR/artefacts/
export PYTHONPATH="./"
export WANDB__SERVICE_WAIT=1000

WANDB_MODE=offline python3 pl_training.py \
    --max_epochs 100 \
    --train_bs 2 \
    --test_bs 2 \
    --ruche \
    --no-use_feature_extractor \
    --task="Conditional_Diffusion" \
    --mix_precision \
    --data_folder="control" \
    --one_batch \





