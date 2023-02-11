from data_file.processing import Dataset
from model.model import Model
import tensorflow as tf
from parsing import parse_args, default_config
import wandb
import wandb_params
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint, WandbEvalCallback
import time


def final_metrics(learn):
    "Log latest metrics values"
    scores = learn.validate()
    metric_names = ["final_loss"] + [f"final_{x.name}" for x in learn.metrics]
    final_results = {metric_names[i]: scores[i] for i in range(len(scores))}
    for k, v in final_results.items():
        wandb.summary[k] = v


def train(config):
    tf.random.set_seed(config.seed)
    t = time.localtime(time.time())
    endian_path = "big_endian/" if config.big_endian else "little_endian/"
    if config.wandb:
        run = wandb.init(
            project=wandb_params.WANDB_PROJECT,
            job_type="train",
            config=config,
            name=config.model
            + "_img_size_"
            + str(config.img_size)
            + "_batchsize_"
            + str(config.batch_size)
            + "_nbepochs_"
            + str(config.epochs)
            + "_lr_"
            + str(config.learning_rate)
            + "_"
            + str(t.tm_mday)
            + "d"
            + str(t.tm_hour)
            + "h",
        )

    config = wandb.config if config.wandb else config
    gpus = tf.config.list_logical_devices("GPU") if len(tf.config.list_physical_devices("GPU")) > 0 else 1
    print(f"Generating sample  with batch_size = {config.batch_size * len(gpus)}")
    dataset = Dataset(
        height=config.img_size,
        width=config.img_size,
        batch_size=config.batch_size * len(gpus),
        big_endian = config.big_endian
    )
    dataset.setup()
    train_ds, valid_ds, test_ds = dataset.train_ds, dataset.valid_ds, dataset.test_ds
    print("Sample Generated!")
    print("Num GPUs Available:", len(gpus))
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
        print("Creating the model ...")
        model = Model(config.model, config.img_size, config.learning_rate).build_model()
        print("Model Created!")

    print("Start Training")
    if config.wandb:
        model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=config.epochs,
            verbose=1,
            callbacks=[
                WandbMetricsLogger(),
                WandbModelCheckpoint(
                    filepath=config.saving_path + endian_path + "config.model{val_loss:.4f}",
                    monitor="generator_loss",
                    mode="min",
                    save_weights_only=True,
                    save_best_only=True,
                    verbose=1,
                ),
            ],
        )
    else:
        model.fit(
            train_ds.take(5),
            validation_data=valid_ds.take(5),
            epochs=config.epochs,
            verbose=1,
        )
    model.save(config.saving_path + endian_path + config.model)
    print("Training Done!")


if __name__ == "__main__":
    parse_args()
    train(default_config)
