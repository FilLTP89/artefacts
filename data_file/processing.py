import tensorflow as tf
import numpy as np
from glob import glob
import re
import os
from sklearn.model_selection import train_test_split
from CBCT_preprocess import read_raw
from visualize import visualize_from_dataset
import h5py
from sklearn.utils import shuffle

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


class Dataset:
    def __init__(
        self,
        path: str = "./data/",
        width: int = 512,
        height: int = 512,
        batch_size: int = 32,
        saving_format: str = None,
        train_saving_path: str = "train/",
        test_saving_path: str = "test/",
        seed: int = 42,
        big_endian: bool = True,
        shuffle = False
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
        self.big_endian = big_endian
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
                glob(os.path.join(self.path, "No_metal/acquisition_" + str(i) + "/*")),
                key=lambda x: [
                    int(c) if c.isdigit() else c for c in re.split(r"(\d+)", x)
                ],
            )
            for i in range(11)
        ]
        high_metal_folder = [
            sorted(
                glob(
                    os.path.join(self.path, "High_metal/acquisition_" + str(i) + "/*")
                ),
                key=lambda x: [
                    int(c) if c.isdigit() else c for c in re.split(r"(\d+)", x)
                ],
            )
            for i in range(11)
        ]
        low_metal_folder = [
            sorted(
                glob(os.path.join(self.path, "Low_metal/acquisition_" + str(i) + "/*")),
                key=lambda x: [
                    int(c) if c.isdigit() else c for c in re.split(r"(\d+)", x)
                ],
            )
            for i in range(11)
        ]


        self.no_metal_folder = no_metal_folder
        self.low_metal_folder = low_metal_folder
        self.high_metal_folder = high_metal_folder

        no_metal_list = [item for sublist in no_metal_folder for item in sublist]
        high_metal_list = [item for sublist in high_metal_folder for item in sublist]
        low_metal_list = [item for sublist in low_metal_folder for item in sublist]

        return (
            no_metal_list,
            high_metal_list,
            low_metal_list,
        )

    def load_data(self):
        """
        Generate the training and testing (X,y) couple using the previously generated list
        """
        no_metal_list, high_metal_list, low_metal_list = self.collect_data()

        label = 2 * no_metal_list
        input = high_metal_list + low_metal_list
        X_train, X_test_1, y_train, y_test_1 = train_test_split(
            input, label, test_size=(2 / 11), random_state=self.seed, shuffle=self.shuffle
        )  # 8 acquisition for training
        X_test, X_valid, y_test, y_valid = train_test_split(
            X_test_1, y_test_1, test_size=0.5, random_state=self.seed, shuffle=self.shuffle
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
            x = x.decode("utf-8")
            y = y.decode("utf-8")
            with_artefact = read_raw(
                x,
                image_size=(self.original_height, self.original_width),
                big_endian=self.big_endian,
            )
            without_artefact = read_raw(
                y,
                image_size=(self.original_height, self.original_width),
                big_endian=self.big_endian,
            )
            return with_artefact, without_artefact

        input, label = tf.numpy_function(f, [x, y], [tf.float32, tf.float32])
        input = tf.expand_dims(input, axis=-1)  # (Height,Width) -> (Height,Width,1)
        label = tf.expand_dims(label, axis=-1)  # (Height,Width) -> (Height,Width,1)
        input.set_shape([self.width, self.height, 1])
        label.set_shape([self.width, self.height, 1])

        input = tf.image.resize(input, [self.width, self.height])
        label = tf.image.resize(label, [self.width, self.height])
        return input, label

    def tf_dataset(self, x, y):
        """
        TO DO : Understand the buffer size
        TO DO : Add other parameters (maybe look on youtube)
        """
        ds = tf.data.Dataset.from_tensor_slices(
            (x, y)
        )  # Create a tf.data.Dataset from the couple
        ds = ds.map(
            self.preprocess
        )  # apply the processing function on the couple (from path of raw image to sinongram)
        ds = ds.batch(self.batch_size)  # Batch the couple into batch of couple
        # ds = ds.prefetch(buffer_size=1024)
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
        if self.saving_format == ("hdf5" or "h5"):
            with h5py.File(f"{self.path}/save/train_dataset.h5", "w") as f:
                dset = f.create_dataset("data", (len(self.train_ds),), dtype="i")
                for i, data in enumerate(self.train_ds):
                    dset[i] = data
            with h5py.File(f"{self.path}/save/test_dataset.h5", "w") as f:
                dset = f.create_dataset("data", (len(self.test_ds),), dtype="i")
                for i, data in enumerate(self.test_ds):
                    dset[i] = data
        else:
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
    
    def load_single_acquisition(self, acquistion_number = 1, low = False):
        no_metal_folder = self.no_metal_folder[acquistion_number]
        metal_folder = self.low_metal_folder[acquistion_number] if low else self.high_metal_folder[acquistion_number]
        ds = self.tf_dataset(metal_folder, no_metal_folder)
        return ds


if __name__ == "__main__":
    print("Generating sample ....")
    dataset = Dataset(path="../data/", batch_size=1, big_endian=True)
    dataset.setup()
    train_ds, valid_ds, test_ds = dataset.train_ds, dataset.valid_ds, dataset.test_ds
    """
    print("Sample Generated!")
    for x, y in train_ds.take(1):
        for i in range(8):
            visualize_from_dataset(
                x[i],
                y[i],
                big_endian=dataset.big_endian,
                brightness_fact=4,
            )
    """
    acquisition = dataset.load_single_acquisition(1)
    print(len(acquisition))