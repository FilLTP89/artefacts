import tensorflow as tf
from data_file.processing import Dataset
from data_file.visualize import save_image_result


def test():
    print("Generate model...")
    model = tf.keras.models.load_model("model/saved_models/ResUnet.h5")
    print("Model generated!")
    dataset = Dataset(height=256, width=256, batch_size=8)
    dataset.setup()
    dataset.test_ds = dataset.test_ds.take(1)
    print("Start testing...")
    for x, y in dataset.test_ds:
        preds = model.predict(x)
        for i in range(8):
            save_image_result(x[i], preds[i], y[i], name=f"test_{i}")
    print("Sample images saved!")


if __name__ == "__main__":
    test()
