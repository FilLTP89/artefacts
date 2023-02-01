import tensorflow as tf
from data_file.processing import Dataset
from data_file.utils import save_file
from model.metrics import ssim


def load_model():
    model = tf.keras.models.load_model(
        "model/saved_models/ResUnet.h5", custom_objects={"ssim": ssim}
    )
    return model


def test(big_endian=True):
    model = load_model()
    dataset = Dataset(height=512, width=512, batch_size=32, big_endian=big_endian)
    dataset.setup()
    train_ds, valid_ds, test_ds = dataset.train_ds, dataset.valid_ds, dataset.test_ds
    print("Evaluating on training set")
    print(model.evaluate(train_ds))
    print("Evaluating on validation set")
    print(model.evaluate(valid_ds))
    print("Evaluating on test set")
    print(model.evaluate(test_ds))


def generate_image():
    print("Generate model...")
    model = tf.keras.models.load_model(
        "model/saved_models/ResUnet.h5", custom_objects={"ssim": ssim}
    )
    print("Model generated!")
    dataset = Dataset(height=512, width=512, batch_size=32, big_endian=True)
    dataset.setup()
    train_ds, valid_ds, test_ds = dataset.train_ds, dataset.valid_ds, dataset.test_ds

    print("Generating test images...")
    batch = 0
    for x, y in dataset.test_ds:
        preds = model(x)
        for i in range(8):
            save_file(x[i], preds[i], y[i], name=f"test_batch{batch}_sample_{i}")
        batch += 1
    print("Generating train images...")
    """ for x, y in dataset.train_ds:
        preds = model(x)
        for i in range(8):
            save_file(x[i], preds[i], y[i], name=f"train_batch{batch}_sample_{i}")
        batch += 1
    print("Generating valid images...")
    for x, y in valid_ds.train_ds:
        preds = model(x)
        for i in range(8):
            save_file(x[i], preds[i], y[i], name=f"valid_batch{batch}_sample_{i}")
        batch += 1 """


if __name__ == "__main__":
    # generate_image()
    test()
