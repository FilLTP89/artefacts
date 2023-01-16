import tensorflow as tf
import numpy as np
from glob import glob
import re
import os
from sklearn.model_selection import train_test_split
from CBCT_preprocess import open_raw
from visualize import visualize_from_datset
import h5py

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
        test_saving_path: str = "data/",
    ) -> None:

        self.path = path
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.saving_format = saving_format
        self.train_saving_path = path + train_saving_path
        self.test_saving_path = path + test_saving_path

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

        sorted_folder_1_name = sorted(
            glob(os.path.join(self.path, "1" + "/*")),
            key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", x)],
        )

        sorted_folder_2_name = sorted(
            glob(os.path.join(self.path, "2" + "/*")),
            key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", x)],
        )

        sorted_folder_4_name = sorted(
            glob(os.path.join(self.path, "4" + "/*")),
            key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", x)],
        )

        return sorted_folder_1_name, sorted_folder_2_name, sorted_folder_4_name

    def load_data(self):
        """
        Generate the training and testing (X,y) couple using the previously generated list
        """
        f1, f2, f4 = self.collect_data()
        label = f1 + f1
        input = f2 + f4

        X_train, X_test = train_test_split(
            input, test_size=0.2, random_state=1, shuffle=True
        )

        y_train, y_test = train_test_split(
            label, test_size=0.2, random_state=1, shuffle=True
        )

        return (X_train, y_train), (X_test, y_test)

    def preprocess(self, x, y):
        """
        TO DO : store the theta parameter somewhere
        TO DO : documentation for this function

        """

        def f(x, y):
            x = x.decode("utf-8")
            y = y.decode("utf-8")
            with_artefact = open_raw(
                x, imageHeight=self.height, imageWidth=self.width
            ).astype(np.float32)
            without_artefact = open_raw(
                y, imageHeight=self.height, imageWidth=self.width
            ).astype(np.float32)
            return with_artefact, without_artefact

        input, label = tf.numpy_function(f, [x, y], [tf.float32, tf.float32])
        input = tf.expand_dims(input, axis=-1)  # (400,400) -> (400,400,1)
        label = tf.expand_dims(label, axis=-1)  # (400,400) -> (400,400,1)
        input.set_shape([self.width, self.height, 1])
        label.set_shape([self.width, self.height, 1])
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
        test_ds = self.tf_dataset(self.X_test, self.y_test)
        return train_ds, test_ds

    def setup(self):
        """
        Generate the different train and test sample either as array or tf.data.Dataset
        """
        (self.X_train, self.y_train), (self.X_test, self.y_test) = self.load_data()
        self.train_ds, self.test_ds = self.load_dataset()

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

    def load(self, path):
        """
        TO DO : Loading h5 file.
        """
        dataset = tf.data.Dataset.load(path)
        return dataset


if __name__ == "__main__":
    print("Generating sample ....")
    dataset = Dataset(path="../data/", batch_size=1)
    dataset.setup()
    train_ds = dataset.train_ds
    print("Sample Generated!")
    for x, y in train_ds.take(1):
        print(x.shape)
        # visualize_from_datset(x[0], y[0])

    """ print("Saving dataset.... ")
    dataset.save()
    print("Dataset saved!") """
