import tensorflow as tf
from data_file.processing import Dataset
from data_file.utils import save_file


def test():
    print("Generate model...")
    model = tf.saved_model.load("model/saved_models/10")
    print("Model generated!")
    dataset = Dataset(height=512, width=512, batch_size=8)
    dataset.setup()
    dataset.test_ds = dataset.test_ds.take(1)
    print("Start testing...")
    for x, y in dataset.test_ds:
        preds = model(x)
        for i in range(8):
            save_file(x[i], preds[i], y[i], name=f"test_{i}")
    print("Sample images saved!")


if __name__ == "__main__":
    test()
