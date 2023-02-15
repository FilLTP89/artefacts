import tensorflow as tf
from data_file.processing import Dataset
from data_file.utils import save_file
from model.metrics import ssim
from model.model import Model
from model.metrics import ssim, psnr, mae, rmse


def load_model(model_path=None):
    """
    model = Model("MedGAN", vgg_whole_arc=True).build_model()
    model.build(input_shape=(None, 512, 512, 1))
    model.load_weights(
        "model/saved_models/MedGAN/bi_endian/MedGAN09/model.ckpt"
    ).expect_partial()
    """
    model = tf.keras.models.load_model(
        "model/saved_models/MedGAN/big_endian/blazing-rose-13/08/"
    )
    model.compile()
    return model


def test(big_endian=False, model_name="ResUnet"):
    gpus = tf.config.list_logical_devices("GPU")
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
        model = load_model()
    dataset = Dataset(height=512, width=512, batch_size=32, big_endian=big_endian)
    dataset.setup()
    train_ds, valid_ds, test_ds = dataset.train_ds, dataset.valid_ds, dataset.test_ds
    # print("Evaluating on training set")
    # print(model.evaluate(train_ds))
    print("Evaluating on validation set")
    print(model.evaluate(valid_ds))
    print("Evaluating on test set")
    print(model.evaluate(test_ds))


def generate_image():
    print("Generate model...")
    model = load_model()
    model = model.generator
    print("Model generated!")

    dataset = Dataset(height=512, width=512, batch_size=32, big_endian=False)
    dataset.setup()
    batch = 0
    print("Generating train images...")
    for batch, (x, y) in enumerate(dataset.train_ds.take(1)):
        preds = model(x)
        for i in range(8):
            save_file(
                x[i],
                preds[i],
                y[i],
                name=f"train_batch{batch}_sample_{i}",
                big_endian=False,
            )


def test_metrics():
    model = load_model()
    model = model.generator
    dataset = Dataset(height=512, width=512, batch_size=32, big_endian=True)
    dataset.setup()
    train_ds, valid_ds, test_ds = dataset.train_ds, dataset.valid_ds, dataset.test_ds
    ssim_v = 0
    psnr_v = 0
    mae_v = 0
    rmse_v = 0
    for x, y in test_ds.take(len(test_ds)):
        preds = model(x)
        ssim_v += ssim(y, preds)
        psnr_v += psnr(y, preds)
        mae_v += mae(y, preds)
        rmse_v += rmse(y, preds)
    print("SSIM: ", ssim_v / len(test_ds))
    print("PSNR: ", psnr_v / len(test_ds))
    print("MAE: ", mae_v / len(test_ds))
    print("RMSE: ", rmse_v / len(test_ds))


if __name__ == "__main__":
    test_metrics()
    # test(model_name="Baseline")
