from data_file.processing import Dataset
from data_file.processing_vgg import VGGDataset
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

def fit_model(model, config, train_ds, valid_ds, test_ds):
    callbacks = []
    endian_path = "big_endian/" if config.big_endian else "low_endian/"
    model.compute_output_shape(input_shape=(None, 512, 512, 1))
    if config.wandb :
        callbacks = [
            WandbMetricsLogger(),
            WandbModelCheckpoint(filepath=config.saving_path + endian_path + config.run_name +"/{epoch:02d}/")] 
    model.fit(
                train_ds,
                validation_data=valid_ds,
                epochs=config.epochs,
                callbacks=callbacks,
        
            )
    #tf.saved_model.save(model,config.saving_path + endian_path + config.model+ "/")
    model.evaluate(test_ds)

def initalize_project_name(config):
    project_name = "MedGAN" if config.model == "MedGAN" else "VGG19"
    project_name += "_big_endian" if config.big_endian else "_low_endian"
    return project_name


def train(config):
    tf.random.set_seed(config.seed)
    t = time.localtime(time.time())   
    if config.wandb:
        run = wandb.init(
            project=initalize_project_name(config),
            job_type="train",
            config=config,
        )

    config = wandb.config if config.wandb else config
    gpus = tf.config.list_logical_devices("GPU") if len(tf.config.list_physical_devices("GPU")) > 0 else 1
    print(f"Generating sample  with batch_size = {config.batch_size * len(gpus)}")
    if config.model == "VGG19":
        dataset = VGGDataset(
            height=config.img_size,
            width=config.img_size,
            batch_size=config.batch_size * len(gpus),
            big_endian=config.big_endian,
        )
    else:
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
    if config.one_batch_training :
        fit_model(model, config, train_ds.take(1), valid_ds.take(1), test_ds.take(1))
    else:
        fit_model(model, config, train_ds, valid_ds, test_ds)
    print("Training Done!")


if __name__ == "__main__":
    parse_args()
    train(default_config)
