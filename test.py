import tensorflow as tf
from data_file.processing import Dataset
from data_file.processing_dicom import DicomDataset
from data_file.utils import save_file
from model.metrics import ssim
from model.MedGAN import MEDGAN
from model.metrics import ssim, psnr, mae, rmse


def load_model(model_path=None):
    model = tf.keras.models.load_model(
        "model/saved_models/MedGAN/big_endian/heartfelt-etchings-23/20"
    )
    model.compile()
    return model


def load_model_with_weights(
    model_path="model/saved_models/MedGAN/dicom/peachy-breeze-4/20/model.ckpt",
):
    model = MEDGAN()
    model.build(input_shape=(None, 512, 512, 1))
    model.save_weights(model_path)
    model.compile()
    return model


def test(big_endian=False, model_name="ResUnet"):
    gpus = tf.config.list_logical_devices("GPU")
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
        model = load_model()
    dataset = Dataset(height=512, width=512, batch_size=32, big_endian=big_endian)
    dataset.setup()
    _, valid_ds, test_ds = dataset.train_ds, dataset.valid_ds, dataset.test_ds
    # print("Evaluating on training set")
    # print(model.evaluate(train_ds))
    print("Evaluating on validation set")
    print(model.evaluate(valid_ds))
    print("Evaluating on test set")
    print(model.evaluate(test_ds))


def generate_image():
    print("Generate model...")
    model = load_model_with_weights()
    model = model.generator
    print("Model generated!")

    dataset = DicomDataset(height=512, width=512, batch_size=32)
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
            )


def test_metrics():
    model = load_model()
    model = model.generator
    dataset = Dataset(height=512, width=512, batch_size=32, big_endian=True)
    dataset.setup()
    train_ds, valid_ds, test_ds = dataset.train_ds, dataset.valid_ds, dataset.test_ds
    model_ssim, model_psnr, model_mae, model_rmse = 0, 0, 0, 0
    baseline_ssim, baseline_psnr, baseline_mae, baseline_rmse = 0, 0, 0, 0
    for x, y in test_ds.take(len(test_ds)):
        preds = model(x)
        model_ssim += ssim(y, preds)
        model_psnr += psnr(y, preds)
        model_mae += mae(y, preds)
        model_rmse += rmse(y, preds)

        baseline_ssim += ssim(y, x)
        baseline_psnr += psnr(y, x)
        baseline_mae += mae(y, x)
        baseline_rmse += rmse(y, x)

    print("Model SSIM: ", model_ssim / len(test_ds))
    print("Model PSNR: ", model_psnr / len(test_ds))
    print("Model MAE: ", model_mae / len(test_ds))
    print("Model RMSE: ", model_rmse / len(test_ds))

    print("Baseline SSIM: ", baseline_ssim / len(test_ds))
    print("Baseline PSNR: ", baseline_psnr / len(test_ds))
    print("Baseline MAE: ", baseline_mae / len(test_ds))
    print("Baseline RMSE: ", baseline_rmse / len(test_ds))


if __name__ == "__main__":
    # test_metrics()
    # test(model_name="Baseline")
    generate_image()
