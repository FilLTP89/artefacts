import os
import re
import pydicom as dicom
import random
import torch
from torch.utils.data import Dataset
import numpy as np
from glob import glob
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt 



def create_dataset(
        path = "datav2/protocole_1/",
        control = True,
        nb_folder = 5,
        dcm = True):

    control = "control" if control else "fracture"
    dcm = "dcm" if dcm else "raw"    


    input_name = [path + f"{control}/{i}/{dcm}/" + "Input/" for i in range(1, nb_folder + 1)]
    target_name = [path + f"{control}/{i}/{dcm}/" + "Target/" for i in range(1, nb_folder + 1)]
    
    target_categories = os.listdir(target_name[0])[0] # control_high
    input_categories = os.listdir(input_name[0])

    input_folders = []
    target_folders = []
    for i in range(nb_folder):
        input_folders.append([input_name[i] + input_categories[j] + "/" for j in range(len(input_categories))])
        target_folders.append([target_name[i] + target_categories + "/"  ])

    target_folders = [item for sublist in target_folders for item in sublist]
    ds = []
    for input_folder,target_folder in zip(input_folders,target_folders):
        input = [[] for i in range(len(input_folder))]
        target_files = sorted(glob(target_folder + f"*.dcm"),
                key=lambda x: [
                    int(c) if c.isdigit() else c for c in re.split(r"(\d+)", x)
                ])
        
        for idx,folder in enumerate(input_folder): 
            input_files = sorted(glob(folder + f"*.dcm"),
                    key=lambda x: [
                        int(c) if c.isdigit() else c for c in re.split(r"(\d+)", x)
                    ])
            try :
                for i_files,t_files in zip(input_files,target_files):
                    input[idx].append((i_files,t_files))
            except:
                pass
        ds.append(input)
    unsqueezed_ds = [item for sublist in ds for item in sublist for item in item]
    return unsqueezed_ds


def sort_key(filename):
    """ Helper function to generate sorting key for filenames with numbers. """
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", filename)]

def gptcreate_dataset(path="datav2/protocole_1/", control=True, nb_folder=5, dcm=True):
    """
    Create a dataset from the specified directory structure.

    Args:
        path (str): The base path for the dataset.
        control (bool): Whether to use 'control' or 'fracture' subdirectories.
        nb_folder (int): Number of folders to process.
        dcm (bool): Whether to use 'dcm' or 'raw' subdirectories.

    Returns:
        list: A list of tuples containing paired input and target files.
    """
    control_folder = "control" if control else "fracture"
    data_format = "dcm" if dcm else "raw"

    input_dirs = [f"{path}{control_folder}/{i}/{data_format}/Input/" for i in range(1, nb_folder + 1)]
    target_dirs = [f"{path}{control_folder}/{i}/{data_format}/Target/" for i in range(1, nb_folder + 1)]

    try:
        target_category = os.listdir(target_dirs[0])[0]  # Only control_high or fracture_high
        input_categories = os.listdir(input_dirs[0])
    except IndexError:
        raise ValueError("The directory structure is not as expected or directories are empty.")

    input_folders = [[f"{input_dir}{category}/" for category in input_categories] for input_dir in input_dirs]
    target_folders = [[f"{target_dir}{target_category}/"] for target_dir in target_dirs]

    dataset = []
    for input_dir_list, target_dir in zip(input_folders, target_folders):
        input_files_per_category = [
            sorted(glob(f"{input_dir}*.dcm"), key=sort_key) for input_dir in input_dir_list
        ]
        target_files = sorted(glob(f"{target_dir[0]}*.dcm"), key=sort_key)

        for input_files in input_files_per_category:
            paired_files = list(zip(input_files, target_files))
            dataset.extend(paired_files)
    return dataset



class Datav2Dataset(Dataset):
    def __init__(self,
                 folder = "datav2/protocole_1/",):
        
        self.folder = gptcreate_dataset(folder)

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, idx):
        input_path, target_path = self.folder[idx]

        input = np.array(dicom.dcmread(target_path).pixel_array, dtype=np.float32)
        target = np.array(dicom.dcmread(input_path).pixel_array, dtype=np.float32)
        return input, target

    def visualize_random(self):
        idx = random.randint(0, len(self) - 1)
        input_name, target_name = self.folder[idx]
        input_name = input_name.split("/")[-2] + "/" + input_name.split("/")[-1]
        target_name = target_name.split("/")[-2] + "/" + target_name.split("/")[-1]
        input, target = self[idx]
        print(f"Input shape: {input.shape}, Target shape: {target.shape}")
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(input, cmap="gray")
        axs[0].set_title(input_name)
        axs[1].imshow(target, cmap="gray")
        axs[1].set_title(target_name)
        plt.show()


if __name__ == "__main__":
    ds = Datav2Dataset()
    ds.visualize_random()
