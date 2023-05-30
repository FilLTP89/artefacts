import tensorflow as tf
from data_file.processing import Dataset
from data_file.processing_dicom import DicomDataset
from data_file.utils import save_file
from model.metrics import ssim
from model.MedGAN import MEDGAN
from model.metrics import ssim, psnr, mae, rmse


def best_model_path(model_name):
    if model_name == 'MedGAN':
        return "model/saved_models/MedGAN/big_endian/heartfelt-etchings-23/20"
    elif model_name == 'ResUnet':
        return "model/saved_models/ResUnet/big_endian/heartfelt-etchings-23/20"
    elif model_name == "DeepMar":
        return "model/saved_models/DeepMar/big_endian/heartfelt-etchings-23/20"
    else:
        return ValueError("Model name not recognized")

def load_model(
    model_path="model/saved_models/MedGAN/big_endian/heartfelt-etchings-23/20",
):
    model = tf.keras.models.load_model(model_path)
    model.compile()
    return model


def load_model_with_weights(
    model_path="model/saved_models/MedGAN/dicom/serene-field-24/20/model.ckpt",
):
    
    gpus = (
        tf.config.list_logical_devices("GPU")
        if len(tf.config.list_physical_devices("GPU")) > 0
        else 1
    ) 
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():  
        model = MEDGAN()
        model.build(input_shape=(None, 512, 512, 1))
        model.load_weights(model_path).expect_partial()
        model.compile()
    return model


def test(big_endian=False, model_name="ResUnet"):
    gpus = tf.config.list_logical_devices("GPU")
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
        model = load_model_with_weights()
    dataset = DicomDataset(height=512, width=512, batch_size=32, big_endian=big_endian)
    dataset.setup()
    _, valid_ds, test_ds = dataset.train_ds, dataset.valid_ds, dataset.test_ds
    # print("Evaluating on training set")
    # print(model.evaluate(train_ds))
    print("Evaluating on validation set")
    print(model.evaluate(valid_ds))
    print("Evaluating on test set")
    #print(model.evaluate(test_ds))


def generate_image(dicom = True):
    print("Generate model...")
    model = load_model_with_weights()
    model = model.generator
    print("Model generated!")

    dataset = DicomDataset(height=512, width=512, batch_size=32) if dicom else Dataset(height=512, width=512, batch_size=32)
    dataset.setup()
    print("Generating train images...")
    for batch, (x, y) in enumerate(dataset.train_ds.take(6)):
        preds = model(x)
        for i in range(8):
            save_file(
                x[i], preds[i], y[i], name=f"train/train_batch{batch}_sample_{i}", dicom=True
            )
    print("Generating valid images...")
    for batch, (x, y) in enumerate(dataset.valid_ds.take(6)):
        preds = model(x)
        for i in range(8):
            save_file(
                x[i], preds[i], y[i], name=f"valid/valid_batch{batch}_sample_{i}", dicom=True
            )
    print("Finish generating images!")

def test_single_acquistion(dicom = True, acquisition_number = 1,batch_size = 32):
    model = load_model_with_weights()
    dataset = DicomDataset(height=512, width=512, batch_size=batch_size, shuffle= False) if dicom else Dataset(height=512, width=512, batch_size=32)
    dataset.setup()
    acquisition = dataset.load_single_acquisition(acquistion_number=acquisition_number)
    file = 0
    for _, (x, y) in enumerate(acquisition):
        preds = model(x)
        for i in range(batch_size):
            save_file(
                x[i], preds[i], y[i], name=f"acquisition_{acquisition_number}/{file}", dicom=True
            )
            file +=1
    

def test_metricsvsBaseline():
    """model = load_model()
    model = model.generator
    """
    model = load_model_with_weights()
    model = model.generator

    dataset = DicomDataset(height=512, width=512, batch_size=32, big_endian=True)
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
    #generate_image()
    #test_single_acquistion()
    test_metricsvsBaseline()

