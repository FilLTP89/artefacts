from data_file.processing import Dataset
from data_file.processing_vgg import VGGDataset
from data_file.processing_dicom import DicomDataset
from data_file.processing_dicom_vgg import DicomVGGDataset
from model.model import Model
import tensorflow as tf
from parsing import parse_args, default_config
import wandb
import wandb_params
from wandb.keras import WandbMetricsLogger
import time
from data_file.processing_segmentation import SegmentationDataset

"""
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")
""" # Contentn loss goes to +inf if mixed precision is used 

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def final_metrics(learn):
    "Log latest metrics values"
    scores = learn.validate()
    metric_names = ["final_loss"] + [f"final_{x.name}" for x in learn.metrics]
    final_results = {metric_names[i]: scores[i] for i in range(len(scores))}
    for k, v in final_results.items():
        wandb.summary[k] = v


def fit_model(model, config, train_ds, valid_ds, test_ds):
    callbacks = []
    if config.dicom:
        endian_path = ""
    else:
        endian_path = "big_endian/" if config.big_endian else "low_endian/"
    dicom_path = "dicom/" if config.dicom else ""
    print("Saving weights only :", config.save_weights)
    if config.wandb:
        callbacks = [
            tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0),
            WandbMetricsLogger(),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=config.saving_path
                + dicom_path
                + endian_path
                + config._settings.run_name
                + "/{epoch:02d}/model.ckpt",
                save_weights_only=config.save_weights,  # save only the weights
            ),
        ]
    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=config.epochs,
        callbacks=callbacks,
    )
    model.evaluate(valid_ds)
    print("Finished training!")
    model.save(config.saving_path + dicom_path + endian_path)


def initalize_project_name(config):
    if config.dicom:
        project_name = f"{config.model}_dicom"
    else:
        project_name = (
            f"{config.model}_big_endian"
            if config.big_endian
            else f"{config.model}_low_endian"
        )
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
    gpus = (
        tf.config.list_logical_devices("GPU")
        if len(tf.config.list_physical_devices("GPU")) > 0
        else 1
    )
    print(f"Generating sample  with batch_size = {config.batch_size * len(gpus)}")
    if config.model == "VGG19":
        print("VGG19 Dataset")
        dataset = VGGDataset(
            height=config.img_size,
            width=config.img_size,
            batch_size=config.batch_size * len(gpus),
            big_endian=config.big_endian,
        )
    elif config.segmentation:
        print("Semantic Dataset")
        dataset = SegmentationDataset(
            height=config.img_size,
            width=config.img_size,
            batch_size=config.batch_size * len(gpus),
        )
    else:
        dataset = Dataset(
            height=config.img_size,
            width=config.img_size,
            batch_size=config.batch_size * len(gpus),
            big_endian=config.big_endian,
        )


    dataset.setup()
    train_ds, valid_ds, test_ds = dataset.train_ds, dataset.valid_ds, dataset.test_ds
    print("Sample Generated!")
    print("Num GPUs Available:", len(gpus))
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
        print("Creating the model ...")
        model = Model(
            model_name=config.model,
            input_shape=config.img_size,
            learning_rate=config.learning_rate,
            big_endian=config.big_endian,
            pretrained_MedGAN=config.pretrained_MedGAN,
        ).build_model()
        print("Model Created!")

    print("Start Training")
    if config.one_batch_training:
        fit_model(
            model,
            config,
            train_ds.take(1),
            valid_ds.take(1),
            test_ds.take(1),
        )
    else:
        fit_model(model, config, train_ds, valid_ds, test_ds)
    print("Training Done!")


def train_dicom(config):
    tf.random.set_seed(config.seed)
    t = time.localtime(time.time())
    if config.wandb:
        run = wandb.init(
            project=initalize_project_name(config),
            job_type="train",
            config=config,
        )

    config = wandb.config if config.wandb else config
    gpus = (
        tf.config.list_logical_devices("GPU")
        if len(tf.config.list_physical_devices("GPU")) > 0
        else 1
    )
    print(f"Generating sample  with batch_size = {config.batch_size * len(gpus)}")
    if config.model == "VGG19":
        dataset = DicomVGGDataset(
            height=config.img_size,
            width=config.img_size,
            batch_size=config.batch_size * len(gpus),
        )
    else:
        dataset = DicomDataset(batch_size=config.batch_size * len(gpus))
    dataset.setup()
    train_ds, valid_ds, test_ds = dataset.train_ds, dataset.valid_ds, dataset.test_ds
    print("Sample Generated!")
    print("Num GPUs Available:", len(gpus))
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
        print("Creating the model with lr =", config.learning_rate)
        model = Model(
            model_name=config.model,
            input_shape=config.img_size,
            learning_rate=config.learning_rate,
            pretrained_MedGAN=config.pretrained_MedGAN,
            dicom=config.dicom,
        ).build_model()
        print("Model Created!")
    print("Start Training")
    if config.one_batch_training:
        fit_model(model, config, train_ds.take(1), valid_ds.take(1), test_ds.take(1))
    else:
        fit_model(model, config, train_ds, valid_ds, test_ds)


if __name__ == "__main__":
    parse_args()
    train_dicom(default_config) if default_config.dicom else train(default_config)
