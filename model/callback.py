import tensorflow as tf
import matplotlib.pyplot as plt


class Callback(tf.keras.callbacks.Callback):
    def __init__(self, save_path, model, train_ds, test_ds) -> None:
        super().__init__()
        self.save_path = save_path
        self.model = model
        self.train_ds = train_ds
        self.test_ds = test_ds

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            x_train, y_train = tf.expand_dims(self.test_ds.take(1)[0], axis=0)
            x_test, y_test = tf.expand_dims(self.train_ds.take(1)[0], axis=0)
