import tensorflow as tf
import numpy as np
from glob import glob
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from CBCT_preprocess import read_raw
from visualize import visualize_from_dataset
import h5py
import pydicom as dicom
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from skimage.transform import radon

"""
Train VGG as a classifier between the different types of images in order to use it later as a feature extractor
"""


# Try to save to use the save function to save the dataset in folder -> Actually took too much times
# See how much volume they take, and how long we take to load them


class DicomVGGDataset:
    def __init__(
        self,
        path: str = "./data/dicom",
        width: int = 512,
        height: int = 512,
        batch_size: int = 32,
        saving_format: str = None,
        train_saving_path: str = "train/",
        test_saving_path: str = "data/",
        seed: int = 42,
    ) -> None:

        self.path = path
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.saving_format = saving_format
        self.train_saving_path = path + train_saving_path
        self.test_saving_path = path + test_saving_path

        self.original_width = 400
        self.original_height = 400

        self.seed = seed

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
            for i in range(11)
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
            for i in range(11)
        ]
        low_metal_folder = [
            sorted(
                glob(os.path.join(self.path, "low_metal/acquisition_" + str(i) + "/*")),
                key=lambda x: [
                    int(c) if c.isdigit() else c for c in re.split(r"(\d+)", x)
                ],
            )
            for i in range(11)
        ]
        no_metal_list = [item for sublist in no_metal_folder for item in sublist]
        high_metal_list = [item for sublist in high_metal_folder for item in sublist]
        low_metal_list = [item for sublist in low_metal_folder for item in sublist]

        return (no_metal_list, high_metal_list, low_metal_list)

    def load_data(self):
        """
        Generate the training and testing (X,y) couple using the previously generated list
        """
        no_metal_list, high_metal_list, low_metal_list = self.collect_data()
        no_metal_label = np.zeros(
            (len(no_metal_list), 1), dtype=np.int8
        )  # label 0 for no_metal_image
        low_metal_label = (
            np.zeros((len(low_metal_list), 1), dtype=np.int8) + 1
        )  # label 1 for low_metal images
        high_metal_label = (
            np.zeros((len(high_metal_list), 1), dtype=np.int8) + 2
        )  # label 2 for high_metal images

        label = np.concatenate(
            [no_metal_label, high_metal_label, low_metal_label], axis=0
        )
        input = no_metal_list + high_metal_list + low_metal_list
        X_train, X_test_1, y_train, y_test_1 = train_test_split(
            input, label, test_size=(2 / 11), random_state=self.seed, shuffle=True
        )  # 8 acquisition for training
        X_test, X_valid, y_test, y_valid = train_test_split(
            X_test_1, y_test_1, test_size=0.5, random_state=self.seed, shuffle=True
        )  # 1 acquisition for validation & 1 acquisition for testing

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
            x = x.decode("utf-8")
            x = dicom.dcmread(x).pixel_array
            x = np.array(x, dtype=np.float32)
            x = 2 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1 # Normalize the image between -1 and 1
            return x, y

        input, label = tf.numpy_function(f, [x, y], [tf.float32, tf.int8])
        input = tf.expand_dims(input, axis=-1)  # (400,400) -> (400,400,1)
        input.set_shape([self.width, self.height, 1])
        label.set_shape([1])
        input = tf.image.resize(input, [self.width, self.height])
        return input, label

    def tf_dataset(self, x, y):
        """
        TO DO : Understand the buffer size
        """
        y = tf.constant(y)
        ds = tf.data.Dataset.from_tensor_slices(
            (x, y)
        )  # Create a tf.data.Dataset from the couple
        ds = ds.map(
            map_func=self.preprocess,
            num_parallel_calls=tf.data.AUTOTUNE,
        )  # apply the processing function on the couple (from path of raw image to sinongram)
        ds = ds.batch(
            batch_size=self.batch_size,
            num_parallel_calls=tf.data.AUTOTUNE,
        )  # Batch the couple into batch of couple
        # ds.cache() # The first time the dataset is iterated over, its elements will be cached either in the specified file or in memory. Subsequent iterations will use the cached data.
        # Our data are too large to be cached in memory
        ds = ds.prefetch(
            buffer_size=tf.data.AUTOTUNE
        )  # Prefetch the next batch while the current one is being processed
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

    def load(self, path):
        """
        TO DO : Loading h5 file.
        """
        self.train_ds = tf.data.Dataset.load(self.train_saving_path)
        self.test_ds = tf.data.Dataset.load(self.test_saving_path)
        return dataset


if __name__ == "__main__":
    print("Generating sample ....")
    dataset = DicomVGGDataset(path="../data/dicom", batch_size=8)
    dataset.setup()
    train_ds, valid_ds, test_ds = dataset.train_ds, dataset.valid_ds, dataset.test_ds
    print("Sample Generated!")
    print("Training dataset : ", len(train_ds))
    print("Validation dataset : ", len(valid_ds))
    print("Testing dataset : ", len(test_ds))
