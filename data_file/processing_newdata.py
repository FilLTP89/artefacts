import os
import re
import pydicom as dicom
import random
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt 
from torchvision import transforms
import torch.nn as nn
from torch.nn.functional import interpolate
from util import normalize_ct_image, CTImageAugmentation
import pytorch_lightning as pl  
import multiprocessing
import torch

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

def classification_dataset(
        path = "datav2/protocole_1/",
        control = True,
        nb_folder = 5,
        dcm = True):
    control = "control" if control else "fracture"
    dcm = "dcm" if dcm else "raw"    


    ds = create_dataset(path, control, nb_folder, dcm)
    transformed_list = [item for tuple_item in ds for item in tuple_item]
    return transformed_list

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

def load_one_acquisition(path, 
                         control=True, 
                         dcm=True,
                         categorie = "cocrhigh",
                         acquisition = 1):
    dataset = gptcreate_dataset(path, control, 5, dcm)
    acquisition = [item for item in dataset if f"{acquisition}/dcm/Input/{categorie}/" in item[0]]
    return acquisition
    
class ClassificationDataset(Dataset):
    def __init__(self,
                 folder = "datav2/protocole_1/",
                 transform = transforms.Compose([
                    transforms.Resize((512, 512)),
                    ])
            ):
        self.folder = classification_dataset(folder)
        self.transform = transform
        self.augmentation = None
        self.class_dict = {
            "Input":0,
            "Target":1,
        }
        #self.augmentation = CTImageAugmentation()

    def __len__(self):
        return len(self.folder)
    
    def __getitem__(self, idx):
        x = self.folder[idx]
        target_or_input = x.split("/")[-3]
        target = self.class_dict[target_or_input]
        x = dicom.dcmread(x).pixel_array
        x = normalize_ct_image(x)
        x = torch.tensor(x).unsqueeze(0)
        if self.transform:
            x = self.transform(x)
        if self.augmentation:
            x = self.augmentation(x)
        return x, target

class Datav2Dataset(Dataset):
    def __init__(self,
                 folder = "datav2/protocole_1/",
                 transform = transforms.Compose([
                    transforms.Resize((512, 512)),
                    ])
            ):
        
        self.folder = gptcreate_dataset(folder)
        self.transform = transform
        self.augmentation = None
        #self.augmentation = CTImageAugmentation()


    def __len__(self):
        return len(self.folder)

    def __getitem__(self, idx):
        input_path, target_path = self.folder[idx]

        input = np.array(dicom.dcmread(target_path).pixel_array, dtype=np.float32)
        target = np.array(dicom.dcmread(input_path).pixel_array, dtype=np.float32)
        input = normalize_ct_image(input, normalization_type='minmax')
        target = normalize_ct_image(target, normalization_type='minmax')
        input = torch.tensor(input).unsqueeze(0)
        target = torch.tensor(target).unsqueeze(0)
        if self.transform:
            input = self.transform(input)
            target = self.transform(target)
        if self.augmentation:
            input, target = self.augmentation(input, target)
        return input, target

    def visualize_random(self):
        idx = random.randint(0, len(self) - 1)
        input_name, target_name = self.folder[idx]
        input_name = input_name.split("/")[-2] + "/" + input_name.split("/")[-1]
        target_name = target_name.split("/")[-2] + "/" + target_name.split("/")[-1]
        input, target = self[idx]
        input = input.permute(1, 2, 0).squeeze().numpy()
        target = target.permute(1, 2, 0).squeeze().numpy()
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(input, cmap="gray")
        axs[0].set_title(input_name)
        axs[1].imshow(target, cmap="gray")
        axs[1].set_title(target_name)
        plt.show()
    
    def load_one_acquisition(self, idx): 
        pass



class LoadOneAcquisition(Dataset):
    def __init__(self,
                 path = "datav2/protocole_1/",
                 control = True,
                 categorie = "cocrhigh",
                 acquisition = 1,
                 transform = transforms.Compose([
                    transforms.Resize((512, 512)),
                    ]),
                 augmentation = None
                 ) -> None:
        super().__init__()
        self.folder = load_one_acquisition(
            path = path,
            control = control,
            categorie = categorie,
            acquisition = acquisition
        )
        self.control = control
        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.folder)
    
    def __getitem__(self, idx):
        input_path, target_path = self.folder[idx]

        input = np.array(dicom.dcmread(target_path).pixel_array, dtype=np.float32)
        target = np.array(dicom.dcmread(input_path).pixel_array, dtype=np.float32)
        input = normalize_ct_image(input)
        target = normalize_ct_image(target)
        input = torch.tensor(input).unsqueeze(0)
        target = torch.tensor(target).unsqueeze(0)
        if self.transform:
            input = self.transform(input)
            target = self.transform(target)
        if self.augmentation:
            input, target = self.augmentation(input, target)
        return input, target



class Datav2Module(pl.LightningDataModule):
    def __init__(self,
                 folder = "datav2/protocole_1/",
                 dataset_type = Datav2Dataset,
                 train_bs = 1,
                 test_bs = 1,
                 train_ratio = 0.8,
                 img_size = 512,
                 num_workers= multiprocessing.cpu_count(),
                 *args, **kwargs):
        
        self.folder = folder
        self.train_bs = train_bs
        self.test_bs = test_bs
        self.train_ratio = train_ratio
        self.valid_ratio = (1 - train_ratio)/2
        self.test_ration = self.valid_ratio
        self.num_workers = self.get_optimal_num_workers()
        self.dataset_type = dataset_type

    def get_optimal_num_workers(self):
        num_cpus = os.cpu_count()
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            return min(num_cpus, 4 * num_gpus)
        else:
            return min(num_cpus, 8)  # Cap at 8 for CPU-only machines
        
    def setup(self, stage = None):
        self.dataset = self.dataset_type(self.folder)
        total = len(self.dataset)
        train_size = int(self.train_ratio * total)
        valid_size = int(self.valid_ratio * total)
        test_size = total - train_size - valid_size
        self.train_ds, self.valid_ds, self.test_ds = torch.utils.data.random_split(self.dataset, [train_size, valid_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_ds, 
                          batch_size=self.train_bs, 
                          num_workers=self.num_workers,
                          shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.valid_ds, 
                          batch_size=self.test_bs, 
                          num_workers=self.num_workers,
                          shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, 
                          batch_size=self.test_bs, 
                          num_workers=self.num_workers,
                          shuffle=False)
    
    def combined_dataloader(self):
        return DataLoader(self.dataset, 
                          batch_size=self.train_bs, 
                          num_workers=self.num_workers,
                          shuffle=True)




if __name__ == "__main__":

    acq = load_one_acquisition(
        path = "datav2/protocole_1/",
        control = True,
        categorie="fibralowmetal",
        acquisition=3
    ) 
    print(acq[:5])
    ds = LoadOneAcquisition(
        path = "datav2/protocole_1/",
        control = True,
        categorie="fibralowmetal",
        acquisition=3
    )
    print(len(ds))
    
    """ ds = ClassificationDataset()
    print(len(ds))
    module = Datav2Module(train_bs=3,
                          dataset_type=ClassificationDataset)
    module.setup()
    train_ds = module.train_dataloader()
    print(len(train_ds))
    for idx, (input, target) in enumerate(train_ds):
        print(input.shape, target.shape)
        x0,y0 = input[0], target[0]
        print(x0)
        print(y0)
        break
    """

    """
     load_one_acquisition(
        path = "datav2/protocole_1/",
        control = True,
        nb_folder = 5,
        dcm = True
    )
   
    ds = Datav2Dataset()
    x,y = ds[0]
    z = x.squeeze()
    for i in z:
        for j in i:
            if j != 0:
                print(j)
    ds.visualize_random()
    module = Datav2Module(train_bs=1)
    module.setup()
    train_ds = module.train_dataloader()
    print(len(module.dataset))
    for idx, (input, target) in enumerate(train_ds):
        print(input.shape, target.shape)
        x0,y0 = input[0], target[0]
        print(x0.max(), x0.min(), y0.max(), y0.min())   
        break
    """