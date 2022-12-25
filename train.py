from data_file.processing import Dataset
from model.ResUNET_a_d6 import ResUNet
from model.Unet import Unet
from parsing import parser


if __name__ == "__main__":
    print("Generating sample ...")
    dataset = Dataset(height=256, width=256, batch_size=1)
    dataset.setup()
    train_ds, test_ds = dataset.train_ds, dataset.test_ds
    print("Sample Generated!")
    print("Creating the model ...")
    model = ResUNet((256, 256, 1), 1).build_model()
    # model = Unet()
    print("Model Created!")
    print("Start Training")
    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=1,
        verbose=1,
    )
    print("Training Done!")
