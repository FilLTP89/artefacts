import tensorflow as tf
from model.vgg19 import VGG19
from data_file.processing_vgg import VGGDataset


if __name__ == "__main__":
    model = VGG19()
    model.load_weights("model/saved_models/VGG19/low_endian/cp.ckpt")
    dataset = VGGDataset(
        height=512,
        width=512,
        batch_size=3,
        big_endian =True
    )
    
    dataset.setup()
    _, _, test_ds = dataset.train_ds, dataset.valid_ds, dataset.test_ds
    model.evaluate(test_ds)
