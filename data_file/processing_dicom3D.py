import torch
import numpy as np
import os
import pydicom as dicom
from glob import glob
import re

from sklearn.model_selection import train_test_split
import time


def collect_data(path):
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
            glob(os.path.join(path, "No_metal/acquisition_" + str(i) + "/*")),
            key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", x)],
        )
        for i in range(1, 11)
    ]
    high_metal_folder = [
        sorted(
            glob(os.path.join(path, "High_metal/acquisition_" + str(i) + "/*")),
            key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", x)],
        )
        for i in range(1, 11)
    ]
    low_metal_folder = [
        sorted(
            glob(os.path.join(path, "Low_metal/acquisition_" + str(i) + "/*")),
            key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", x)],
        )
        for i in range(1, 11)
    ]

    label = 2 * no_metal_folder
    input = low_metal_folder + high_metal_folder

    return input, label


class Dicom3DDataset(torch.utils.data.Dataset):
    def __init__(self, inputs_path, labels_path, shape=512) -> None:
        super().__init__()

        self.original_shape = 588
        self.shape = shape
        self.inputs_path = inputs_path
        self.labels_path = labels_path

    def __len__(self):
        return len(self.inputs_path)

    def __getitem__(self, index):

        empty_x = np.zeros((self.original_shape, self.original_shape, 588))
        empty_y = np.zeros((self.original_shape, self.original_shape, 588))

        for j in range(588):

            x = self.inputs_path[index][j]
            y = self.labels_path[index][j]

            x = dicom.dcmread(x).pixel_array
            y = dicom.dcmread(y).pixel_array

            x = np.array(x, dtype=np.float32)
            y = np.array(y, dtype=np.float32)

            x = x / np.max(x)
            y = y / np.max(y)

            empty_x[index, :, :] = x  # channel first for torch
            empty_y[index, :, :] = y

        return empty_x, empty_y


class Dicom3DModule:
    def __init__(self, PATH, batch_size=1) -> None:
        self.PATH = PATH
        self.batch_size = batch_size
        self.inputs_path, self.labels_path = collect_data(self.PATH)

    def train_test_split(self):
        (
            self.train_inputs,
            self.test_inputs,
            self.train_labels,
            self.test_labels,
        ) = train_test_split(
            self.inputs_path, self.labels_path, test_size=0.1, random_state=42
        )

    def train_val_split(self):
        (
            self.train_inputs,
            self.val_inputs,
            self.train_labels,
            self.val_labels,
        ) = train_test_split(
            self.train_inputs, self.train_labels, test_size=0.1, random_state=42
        )

    def train_loader(self):
        self.train_dataset = Dicom3DDataset(
            inputs_path=self.train_inputs, labels_path=self.train_labels
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

    def val_loader(self):
        self.val_dataset = Dicom3DDataset(
            inputs_path=self.val_inputs,
            labels_path=self.val_labels,
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=10
        )

    def test_loader(self):
        self.test_dataset = Dicom3DDataset(
            inputs_path=self.test_inputs, labels_path=self.test_labels
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=10
        )

    def setup(self):
        self.train_test_split()
        self.train_val_split()
        self.train_loader()
        self.val_loader()
        self.test_loader()


if __name__ == "__main__":
    module = Dicom3DModule(PATH="../data/dicom/", batch_size=1)
    module.setup()
    train_loader = module.train_loader

    time_start = time.time()
    for index, data in enumerate(train_loader):
        print(f"batch : {index}, x : {data[0].shape}, y : {data[1].shape}")
    time_end = time.time()
    print(f"Time taken : {time_end - time_start}")
