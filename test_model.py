import tensorflow as tf
from model.vgg19 import VGG19
from model.MedGAN import MEDGAN 
from data_file.processing_vgg import VGGDataset
from data_file.processing import Dataset

if __name__ == "__main__":
    model = MEDGAN(vgg_whole_arc= True)
    model.build(input_shape = (None,512,512,1))
    model.load_weights("model/saved_models/MedGAN/low_endian/MedGAN01/model.ckpt").expect_partial() # not loading optimizer
    model.compile()
    print("Model Loaded")
    dataset = Dataset(
        height=512,
        width=512,
        batch_size=3,
        big_endian =True
    )
    dataset.setup()
    _, valid_ds, test_ds = dataset.train_ds, dataset.valid_ds, dataset.test_ds
    model.compile()
    model.evaluate(valid_ds.take(1))
