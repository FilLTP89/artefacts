import tensorflow as tf
from data_file.processing import Dataset
from data_file.utils import save_file


def test():
    print("Generate model...")
    model = tf.saved_model.load("model/saved_models/sweep/ResUnet_512_0.000514")
    print("Model generated!")
    dataset = Dataset(height=512, width=512, batch_size=32)
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
    for x, y in dataset.train_ds:
        preds = model(x)
        for i in range(8):
            save_file(x[i], preds[i], y[i], name=f"train_batch{batch}_sample_{i}")
        batch += 1
    print("Generating valid images...")
    for x, y in valid_ds.train_ds:
        preds = model(x)
        for i in range(8):
            save_file(x[i], preds[i], y[i], name=f"valid_batch{batch}_sample_{i}")
        batch += 1


if __name__ == "__main__":
    test()
