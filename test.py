import tensorflow as tf
from data_file.processing import Dataset
from data_file.processing_dicom import DicomDataset
from data_file.utils import save_file, save_to_raw
from model.metrics import ssim
from model.MedGAN import MEDGAN
from model.metrics import ssim, psnr, mae, rmse
from data_file.utils import save_to_raw
import numpy as np
import os
from data_file.processing_segmentation import SegmentationDataset 
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
    model_path="model/saved_models/MedGAN/big_endian/vibrant-dawn-3/40",
):
    try:
        model = tf.keras.models.load_model(model_path)
        model.compile()
    except:
        model_path += "/model.ckpt"
        model = MEDGAN()
        model.build(input_shape=(None, 512, 512, 1))
        model.load_weights(model_path).expect_partial()
    return model

def load_segmentation_model(
        model_path = ""
    ):
    return load_model(model_path)

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

def test_single_acquistion(dicom = False,acquisition_number = 1,batch_size = 32, metal_low = True):
    if dicom : 
        model = load_model_with_weights()
        dataset = DicomDataset(height=512, width=512, batch_size=batch_size, shuffle= False) if dicom else Dataset(height=512, width=512, batch_size=32)
        dataset.setup()
        acquisition = dataset.load_single_acquisition(acquistion_number=acquisition_number)
    else :
        model = load_model()
        big_endian = True
        dataset = Dataset(big_endian = True, batch_size=batch_size)
        dataset.setup()
        acquisition = dataset.load_single_acquisition(acquisition_number, low = metal_low)
    d = 0
    """ while os.path.exists(f"generated_images/big_endian/experiment_{d}"):
        d += 1 """
    metal = "metal_low" if metal_low else "metal_high"
    if not os.path.exists(f"generated_images/acquisition_{acquisition_number}/{metal}"):
        os.makedirs(f"generated_images/acquisition_{acquisition_number}/{metal}")
    file = 0
    for _, (x, y) in enumerate(acquisition):
        preds = model(x)
        for i in range(batch_size):
            save_to_raw(
                x[i], preds[i], y[i], name=f"acquisition_{acquisition_number}/{metal}/{file}",
                big_endian=big_endian
                )
            file +=1
            print(f"File : {file} created")
    
def test_metrics(dicom = False, big_endian = True, batch_size = 32, low = False):
    
    print(f" Testing metrics with low metal : {low}")
    psnr_train, psnr_test = [], []
    ssim_train, ssim_test = [], []
    mae_train, mae_test = [], []
    rmse_train, rmse_test = [], []

    original_psnr_train, original_psnr_test = [], []
    original_ssim_train, original_ssim_test = [], []
    original_mae_train, original_mae_test = [], []
    original_rmse_train, original_rmse_test = [], []

    if dicom : 
        model = load_model_with_weights()
        dataset = DicomDataset(height=512, width=512, batch_size=batch_size, shuffle= False) if dicom else Dataset(height=512, width=512, batch_size=32)
        dataset.setup()
    elif big_endian :
        model = load_model()
        print("Model loaded ! ")
        dataset = Dataset(big_endian= True)
        dataset.setup()
        print("Dataset Loaded !")
    for acquisition_number in range(11):
        model_ssim, model_psnr, model_mae, model_rmse = 0, 0, 0, 0
        original_ssim, original_psnr, original_mae, original_rmse = 0, 0, 0, 0
        acquisition = dataset.load_single_acquisition(acquisition_number, low = low)
        for _, (x, y) in enumerate(acquisition):
            preds = model(x)
            model_ssim += ssim(y, preds)
            model_psnr += psnr(y, preds)
            model_mae += mae(y, preds)
            model_rmse += rmse(y, preds)

            original_ssim += ssim(y, x)
            original_psnr += psnr(y, x)
            original_mae += mae(y, x)
            original_rmse += rmse(y, x)

        print("Acquisition number : ", acquisition_number)
        print("Model SSIM: ", model_ssim / len(acquisition))
        print("Model PSNR: ", model_psnr / len(acquisition))
        print("Model MAE: ", model_mae / len(acquisition))
        print("Model RMSE: ", model_rmse / len(acquisition))
        print()

        print("Original SSIM: ", original_ssim / len(acquisition))
        print("Original PSNR: ", original_psnr / len(acquisition))
        print("Original MAE: ", original_mae / len(acquisition))
        print("Original RMSE: ", original_rmse / len(acquisition))
        print()
        if acquisition_number < 2:
            ssim_test.append(model_ssim / len(acquisition))
            psnr_test.append(model_psnr / len(acquisition))
            mae_test.append(model_mae / len(acquisition))
            rmse_test.append(model_rmse / len(acquisition))

            original_ssim_test.append(original_ssim / len(acquisition))
            original_psnr_test.append(original_psnr / len(acquisition))
            original_mae_test.append(original_mae / len(acquisition))
            original_rmse_test.append(original_rmse / len(acquisition))


        else : 
            ssim_train.append(model_ssim / len(acquisition))
            psnr_train.append(model_psnr / len(acquisition))
            mae_train.append(model_mae / len(acquisition))
            rmse_train.append(model_rmse / len(acquisition))

            original_ssim_train.append(original_ssim / len(acquisition))
            original_psnr_train.append(original_psnr / len(acquisition))
            original_mae_train.append(original_mae / len(acquisition))
            original_rmse_train.append(original_rmse / len(acquisition))

        
    
    print(f"Mean MAE on train : {np.mean(mae_train)} STD MAE on train : {np.std(mae_train)}")
    print(f"Mean PSNR on train : {np.mean(psnr_train)} STD PSNR on train : {np.std(psnr_train)}")
    print(f"Mean SSIM on train : {np.mean(ssim_train)} STD SSIM on train : {np.std(ssim_train)}")
    print(f"Mean RMSE on train : {np.mean(rmse_train)} STD RMSE on train : {np.std(rmse_train)}")

    print(f"Mean MAE on test : {np.mean(mae_test)} STD MAE on test : {np.std(mae_test)}")
    print(f"Mean PSNR on test : {np.mean(psnr_test)} STD PSNR on test : {np.std(psnr_test)}")
    print(f"Mean SSIM on test : {np.mean(ssim_test)} STD SSIM on test : {np.std(ssim_test)}")
    print(f"Mean RMSE on test : {np.mean(rmse_test)} STD RMSE on test : {np.std(rmse_test)}")
    
    
    print(f"Mean Original MAE on train : {np.mean(original_mae_train)} STD Original MAE on train : {np.std(original_mae_train)}")
    print(f"Mean Original PSNR on train : {np.mean(original_psnr_train)} STD Original PSNR on train : {np.std(original_psnr_train)}")
    print(f"Mean Original SSIM on train : {np.mean(original_ssim_train)} STD Original SSIM on train : {np.std(original_ssim_train)}")
    print(f"Mean Original RMSE on train : {np.mean(original_rmse_train)} STD Original RMSE on train : {np.std(original_rmse_train)}")

    print(f"Mean Original MAE on test : {np.mean(original_mae_test)} STD Original MAE on test : {np.std(original_mae_test)}")
    print(f"Mean Original PSNR on test : {np.mean(original_psnr_test)} STD Original PSNR on test : {np.std(original_psnr_test)}")
    print(f"Mean Original SSIM on test : {np.mean(original_ssim_test)} STD Original SSIM on test : {np.std(original_ssim_test)}")
    print(f"Mean Original RMSE on test : {np.mean(original_rmse_test)} STD Original RMSE on test : {np.std(original_rmse_test)}")



def test_metricsvsBaseline():
    """model = load_model()
    model = model.generator
    """
    model = load_model_with_weights()
    model = model.generator

    dataset = DicomDataset(height=512, width=512, batch_size=32)
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

def segmentation_generation():
    dataset = SegmentationDataset()
    dataset.setup()
    model = load_segmentation_model()
    for batch, (x, y) in enumerate(dataset.train_ds):
        preds = model(x)
        preds = tf.math.round(preds)

    return  





if __name__ == "__main__":
    # test_metrics()
    # test(model_name="Baseline")
    #generate_image()
    test_single_acquistion(dicom=False, acquisition_number=1, batch_size=1, metal_low = True) # batch_size = 1 to avoid memory error on GPU
    #test_metrics(dicom = False, big_endian = True, batch_size = 32, low=False)


