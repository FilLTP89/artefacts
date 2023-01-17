from data_file.processing import Dataset
from model.model import Model
import tensorflow as tf
from parsing import parse_args, default_config
import wandb
import wandb_params
from wandb.keras import WandbMetricsLogger


def final_metrics(learn):
    "Log latest metrics values"
    scores = learn.validate()
    metric_names = ["final_loss"] + [f"final_{x.name}" for x in learn.metrics]
    final_results = {metric_names[i]: scores[i] for i in range(len(scores))}
    for k, v in final_results.items():
        wandb.summary[k] = v


def train(config):
    if default_config.wandb:
        run = wandb.init(
            project=wandb_params.WANDB_PROJECT,
            job_type="train",
            config=config,
            name=default_config.model
            + "_"
            + str(default_config.img_size)
            + "_"
            + str(default_config.batch_size)
            + "_"
            + str(default_config.epochs)
            + "_"
            + str(default_config.learning_rate),
        )

    config = wandb.config if default_config.wandb else config
    print(f"Generating sample  with batch_size = {config.batch_size * config.gpus}")
    dataset = Dataset(
        height=config.img_size,
        width=config.img_size,
        batch_size=config.batch_size * config.gpus,
    )
    dataset.setup()
    train_ds, test_ds = dataset.train_ds, dataset.test_ds
    print("Sample Generated!")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
    gpus = tf.config.list_logical_devices("GPU")
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
        print("Creating the model ...")
        print(config.model)
        model = Model(config.model, config.img_size, config.learning_rate).build_model()
        print("Model Created!")

    print("Start Training")
    if default_config.wandb:
        model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=config.epochs,
            verbose=1,
            callbacks=[WandbMetricsLogger()],
        )
    else:
        if config.save:
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath="model/saved_models/" + config.run_name + ".h5",
                    save_best_only=True,
                    monitor="val_loss",
                    mode="min",
                )
            ]
            model.fit(
                train_ds,
                validation_data=test_ds,
                epochs=config.epochs,
                verbose=1,
                callbacks=callbacks,
            )
        else:
            model.fit(
                train_ds,
                validation_data=test_ds,
                epochs=config.epochs,
                verbose=1,
            )

    print("Training Done!")


if __name__ == "__main__":
    parse_args()
    train(default_config)
