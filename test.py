import tensorflow as tf
from data_file.processing import Dataset
from data_file.utils import save_file
from model.metrics import ssim
from model.model import Model


def load_model(model_name="ResUnet", model_path=None):
    model = Model("MedGAN", pretrained_vgg=False).build_model()
    model.build(input_shape = (None,512,512,1))
    model.load_weights("model/saved_models/MedGAN/low_endian/MedGAN01/model.ckpt")
    return model


def test(big_endian=False, model_name="ResUnet"):
    gpus = tf.config.list_logical_devices("GPU")
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
        model = load_model(model_name)
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
    model = Model("MedGAN", pretrained_vgg=False).build_model()
    model.build(input_shape = (None,512,512,1))
    model.load_weights("/Users/hugo/DeepLearning/projet_these/artefacts/model/saved_models/low_endian/model.ckpt")
    model = model.generator
    print("Model generated!")

    dataset = Dataset(height=512, width=512, batch_size=32, big_endian=False)
    dataset.setup()
    batch = 0
    print("Generating train images...")
    for x, y in dataset.train_ds.take(1):
        preds = model(x)
        print("t2")
        for i in range(8):
            save_file(x[i], preds[i], y[i], name=f"train_batch{batch}_sample_{i}",big_endian=False)
        batch += 1



if __name__ == "__main__":
    generate_image()
    #test(model_name="Baseline")
