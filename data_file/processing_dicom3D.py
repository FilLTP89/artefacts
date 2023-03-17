import tensorflow as tf
import numpy as np
from glob import glob
import re
import os
from sklearn.model_selection import train_test_split
from visualize import visualize_from_dataset
import h5py
from sklearn.utils import shuffle
import pydicom as dicom
from skimage.transform import radon
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
We know that corresponding image from the different folder 
have the same name and that the input are in folder 2 & 4 while the the 
label are in folder 1.
So we will create couple 
(X = file_from_folder_2 or 4, Y = file_from_folder_1) 
We will exploit that in order to create the training and test couples.
"""


# Try to save to use the save function to save the dataset in folder -> Actually took too much times
# See how much volume they take, and how long we take to load them


class DicomDataset3D:
    def __init__(
        self,
        path: str = "./data/dicom/",
        width: int = 512,
        height: int = 512,
        batch_size: int = 32,
        saving_format: str = None,
        train_saving_path: str = "train/",
        test_saving_path: str = "test/",
        valid_saving_path: str = "valid/",
        seed: int = 1612,
        shuffle=True,
        umax=4095,
        umin=0,
    ) -> None:

        self.path = path
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.saving_format = saving_format
        self.train_saving_path = path + train_saving_path
        self.test_saving_path = path + test_saving_path
        self.valid_saving_path = path + valid_saving_path

        self.original_width = 400
        self.original_height = 400

        self.seed = seed
        self.shuffle = shuffle

    def collect_data(self):
        """
        Create list of path of the raw image from each folder

        Args
        ----

        Returns
        -----
        sorted_folder_1_name(list) : list of the path of all image in folder 1 path
        sorted_folder_2_name(list) : list of the path of all image in folder 2 path
        sorted_folder_4_name(list) : list of the path of all image in folder 4 path
        """
        no_metal_folder = [
            sorted(
                glob(os.path.join(self.path, "no_metal/acquisition_" + str(i) + "/*")),
                key=lambda x: [
                    int(c) if c.isdigit() else c for c in re.split(r"(\d+)", x)
                ],
            )
            for i in range(1, 11)
        ]
        high_metal_folder = [
            sorted(
                glob(
                    os.path.join(self.path, "high_metal/acquisition_" + str(i) + "/*")
                ),
                key=lambda x: [
                    int(c) if c.isdigit() else c for c in re.split(r"(\d+)", x)
                ],
            )
            for i in range(1, 11)
        ]
        low_metal_folder = [
            sorted(
                glob(os.path.join(self.path, "low_metal/acquisition_" + str(i) + "/*")),
                key=lambda x: [
                    int(c) if c.isdigit() else c for c in re.split(r"(\d+)", x)
                ],
            )
            for i in range(1, 11)
        ]

        return (
            no_metal_folder,
            low_metal_folder,
            high_metal_folder,
        )

    def load_data(self):
        """
        Generate the training and testing (X,y) couple using the previously generated list
        """
        no_metal_list, high_metal_list, low_metal_list = self.collect_data()

        label = 2 * no_metal_list
        input = high_metal_list + low_metal_list
        X_train, X_test_1, y_train, y_test_1 = train_test_split(
            input,
            label,
            test_size=(2 / 11),
            random_state=self.seed,
            shuffle=self.shuffle,
        )  # 8 acquisition for training
        X_test, X_valid, y_test, y_valid = train_test_split(
            X_test_1,
            y_test_1,
            test_size=0.5,
            random_state=self.seed,
            shuffle=self.shuffle,
        )  # 1 acquisition for validation & 1 acquisition for testing

        # X_train, y_train = shuffle(X_train, y_train, random_state=self.seed)
        # Train : acquisition 0 to 8
        # Test : acquisition 9
        # Valid : acquisition 10
        return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

    def preprocess(self, x, y):
        """
        TO DO : store the theta parameter somewhere
        TO DO : documentation for this function

        """

        def f(x, y):
            empty_x = np.zeros((588, 588, len(x)))
            empty_y = np.zeros((588, 588, len(y)))
            for index, (acquisition_x, acquisition_y) in enumerate(
                tqdm(zip(x, y), total=len(x))
            ):
                decoded_x = acquisition_x.decode("utf-8")
                decoded_y = acquisition_y.decode("utf-8")

                image_x = dicom.dcmread(decoded_x).pixel_array
                image_y = dicom.dcmread(decoded_y).pixel_array

                image_x = np.array(image_x, dtype=np.float32)
                image_y = np.array(image_y, dtype=np.float32)

                image_x = image_x / 6500
                image_y = image_y / 6500  # 6500 = random  number

                empty_x[:, :, index] = image_x
                empty_y[:, :, index] = image_y
            return empty_x, empty_y

        input, label = tf.numpy_function(f, [x, y], [tf.float32, tf.float32])
        input.set_shape([self.width, self.height, 588])  # channel at the end
        label.set_shape([self.width, self.height, 588])

        input = tf.image.resize(input, [self.width, self.height])
        label = tf.image.resize(label, [self.width, self.height])

        return input, label

    def tf_dataset(self, x, y):
        """
        TO DO : Understand the buffer size
        """
        ds = tf.data.Dataset.from_tensor_slices(
            (x, y)
        )  # Create a tf.data.Dataset from the couple
        ds = ds.map(
            map_func=self.preprocess,
            num_parallel_calls=tf.data.AUTOTUNE,
        )  # apply the processing function on the couple (from path of raw image to sinongram)
        ds = ds.batch(
            self.batch_size,
            num_parallel_calls=tf.data.AUTOTUNE,
        )  # Batch the couple into batch of couple
        # ds = ds.prefetch(buffer_size=1024)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    def load_dataset(self):
        train_ds = self.tf_dataset(self.X_train, self.y_train)
        valid_ds = self.tf_dataset(self.X_valid, self.y_valid)
        test_ds = self.tf_dataset(self.X_test, self.y_test)
        return train_ds, valid_ds, test_ds

    def setup(self):
        """
        Generate the different train and test sample either as array or tf.data.Dataset
        """
        (
            (self.X_train, self.y_train),
            (self.X_valid, self.y_valid),
            (self.X_test, self.y_test),
        ) = self.load_data()
        self.train_ds, self.valid_ds, self.test_ds = self.load_dataset()

    def save(self):
        """
        TO DO : test this method for both h5 and normal format
        """
        """
        Save the train and test dataset in their corresponding path
        """
        self.train_ds.save(self.train_saving_path)
        self.test_ds.save(self.test_saving_path)
        self.valid_ds.save(self.valid_saving_path)

    def load(self):
        """
        TO DO : Loading h5 file.
        """
        self.train_ds = tf.data.Dataset.load(self.train_saving_path)
        self.test_ds = tf.data.Dataset.load(self.test_saving_path)
        self.valid_ds = tf.data.Dataset.load(self.valid_saving_path)


if __name__ == "__main__":
    print("Generating sample ....")
    dataset = DicomDataset3D(path="../data/dicom/", batch_size=3)
    dataset.setup()
    train_ds, valid_ds, test_ds = dataset.train_ds, dataset.valid_ds, dataset.test_ds
    for x, y in train_ds.take(1):
        print(x.shape)
        print(y.shape)
        break
