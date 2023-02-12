import tensorflow as tf
from model.vgg19 import VGG19
from data_file.processing_vgg import VGGDataset


if __name__ == "__main__":
    model = VGG19(classifier_training=True)
    model.build(input_shape = (None,512,512,1))
    model.load_weights("model/saved_models/VGG19/low_endian/cp.ckpt")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=tf.keras.metrics.SparseCategoricalAccuracy(),
            )
    dataset = VGGDataset(
        height=512,
        width=512,
        batch_size=3,
        big_endian =True
    )
    dataset.setup()
    _, valid_ds, test_ds = dataset.train_ds, dataset.valid_ds, dataset.test_ds
    model.evaluate(valid_ds)