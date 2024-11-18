import os
import re
import pydicom as dicom
from tqdm import tqdm
import random
import torch
import torch.nn.functional as F
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
from augmentation import DicomClassificationCollator, DicomPredictionCollator

def select_folder(path,control = True ,category="controlhigh", dcm= True):
    folder = []
    dcm = "dcm" if dcm else "raw"
    control = "control" if control else "fracture"
    for i in range(1,6):
        try:
            folder += os.listdir(os.path.join(path, f"{control}/{i}/{dcm}/Input/{category}"))
        except:
            pass
    return folder

def create_all_dataset(
        path = "datav2/protocole_1/",
        nb_folder = 5,
        dcm = True):
    control = create_dataset(path, control = True, nb_folder = nb_folder, dcm = dcm)
    fracture = create_dataset(path, control = False, nb_folder = nb_folder, dcm = dcm)
    return control +  fracture


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

def gpt_create_all_dataset(path="datav2/protocole_1/", nb_folder=5, dcm=True):
    control = gptcreate_dataset(path, control=True, nb_folder=nb_folder, dcm=dcm)
    fracture = gptcreate_dataset(path, control=False, nb_folder=nb_folder, dcm=dcm)
    return control + fracture


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
    control_folder = "control"
    data_folder = "control" if control else "fracture"
    data_format = "dcm" if dcm else "raw"

    input_dirs = [f"{path}{data_folder}/{i}/{data_format}/Input/" for i in range(1, nb_folder + 1)]
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

def load_one_acquisition(path = "datav2/protocole_1/", 
                         control=True, 
                         dcm=True,
                         categorie = "cocrhigh",
                         acquisition = 1):
    dataset = gptcreate_dataset(path, control=control, dcm=dcm)
    control = "control" if control else "fracture"
    acquisition = [item for item in dataset if f"{acquisition}/dcm/Input/{categorie}" in item[0]]
    return acquisition

def load_all_acquisition(path = "datav2/protocole_1/",
                            control=True,
                            dcm=True,
    ):
    dataset = gptcreate_dataset(path)
    total_path = f"{path}{'control' if control else 'fracture'}/"
    numbers = os.listdir(total_path) 
    categories = os.listdir(total_path + numbers[0] + "/dcm/Input/")
    all_acquisition = []
    for categorie in categories:
        for number in numbers:
            acquisition = [item for item in dataset if f"{number}/dcm/Input/{categorie}" in item[0]]
            all_acquisition.append(acquisition)
    return all_acquisition




class ClassificationDataset(Dataset):
    def __init__(self,
                 folder = "datav2/protocole_1/",
                 data_folder = "complete",
                *args, **kwargs
            ):
        self.data_folder = data_folder    
        self.folder = folder    
        self.create_ds()

        self.augmentation = None
        self.n_class = len(self.category_dict)


    def create_ds(self):
        if self.data_folder == "complete":
            self.folder = gpt_create_all_dataset(self.folder)
            self.folder = [x[0] for x in self.folder]

            control_controlhigh = select_folder(path = self.folder,
                                                control = True,
                                                category="controlhigh")
            control_fibrahigh = select_folder(path = self.folder,
                                                control = True,
                                                category="fibrahigh")
            fracture_controlhigh = select_folder(path =self.folder,
                                                control = False,
                                                category="control_high")
            fracture_fibrahigh = select_folder(path =self.folder,
                                                control = False,
                                                category="fibra_high")
            self.folder = self.folder + control_controlhigh + control_fibrahigh + fracture_controlhigh + fracture_fibrahigh
            self.category_dict ={
                "control_cocrhighmetal" : 0,
                "control_cocrlow" : 1,
                "control_cocrlowmetal" : 2,
                "control_controlhighmetal" : 3,
                "control_controllowmetal" : 4,
                "control_fibrahighmetal" : 5,
                "control_fibralow" : 6,
                "control_fibralowmetal" : 7,
                "control_guttahigh" : 8,
                "control_guttahighmetal" : 9,
                "control_guttalow" : 10,
                "control_huttalowmetal" : 11,
                "fracture_cocr_high":12,
                "fracture_cocr_high_metal":13,
                "fracture_cocr_low":14,
                "fracture_cocr_low_metal":15,
                "fracture_control_high_metal":16,
                "fracture_control_low":17,
                "fracture_control_low_metal":18,
                "fracture_fibra_high_metal":19,
                "fracture_fibra_low":20,
                "fracture_fibra_low_metal":21,
                "fracture_gutta_high":22,
                "fracture_gutta_high_metal":23,
                "fracture_gutta_low":24,
                "fracture_gutta_low_metal":25,
                "control_controlhigh" : 26,
                "control_fibrahigh" : 27,
                "fracture_controlhigh" : 28,
                "fracture_fibrahigh" : 29,
                "control_cocrhigh" : 30,
            }

        elif self.data_folder == "control":
            self.folder = classification_dataset(path = self.path)
            self.category_dict = {
                "control_cocrhigh" : 0,
                "control_cocrhighmetal" : 1,
                "control_cocrlow" : 2,
                "control_cocrlowmetal" : 3,
                "control_controlhighmetal" : 4,
                "control_controllowmetal" : 5,
                "control_fibrahighmetal" : 6,
                "control_fibralow" : 7,
                "control_fibralowmetal" : 8,
                "control_guttahigh" : 9,
                "control_guttahighmetal" : 10,
                "control_guttalow" : 11,
                "control_huttalowmetal" : 12,
           }
        else:
            print("Unrecognized dataset argument")

    def get_name(self, name):
        category = name.split("/")[-2]
        control = name.split("/")[-6]
        name = control + "_" + category   
        name = name.replace(" ","_")
        return name


    def normalize(self, image_array):
        """Simple min-max normalization"""
        min_val = image_array.min()
        max_val = image_array.max()
        return (image_array - min_val) / (max_val - min_val), (min_val, max_val)

    def denormalize(self, normalized_array, original_range):
        """Restore original values"""
        min_val, max_val = original_range
        return normalized_array * (max_val - min_val) + min_val


    def __len__(self):
        return len(self.folder)
    
    def __getitem__(self, idx):
        x = self.folder[idx]
        target_or_input = self.get_name(x)
        target = self.category_dict[target_or_input]

        input_dcm = dicom.dcmread(x)
        input = input_dcm.pixel_array
        input = input.astype(np.float32) 
        input_norm, input_range = self.normalize(input)

        input = torch.tensor(input_norm).unsqueeze(0)

        target = torch.tensor(target).type(torch.LongTensor)
        return input, target


class Datav2Dataset(Dataset):     
    def __init__(self,
                 folder="datav2/protocole_1/",
                 data_folder="complete",
                 augmentation=None,
                 prediction_mode=False,
                 *args, **kwargs):
        if data_folder == "complete":
            self.folder = gpt_create_all_dataset(folder)
        elif data_folder == "control":
            self.folder = gptcreate_dataset(folder, control=True)
        else:
            self.folder = gptcreate_dataset(folder, control=False)
        
        self.prediction_mode = prediction_mode
        self.augmentation = augmentation
        self.n_class = 31

    def normalize(self, image_array):
        """Simple min-max normalization"""
        min_val = image_array.min()
        max_val = image_array.max()
        return (image_array - min_val) / (max_val - min_val), (min_val, max_val)

    def denormalize(self, normalized_array, original_range):
        """Restore original values"""
        min_val, max_val = original_range
        return normalized_array * (max_val - min_val) + min_val

    def __getitem__(self, idx):
        input_path, target_path = self.folder[idx]
        
        # Load input
        input_dcm = dicom.dcmread(input_path)
        input_arr = input_dcm.pixel_array.astype(np.float32)
        
        # Load target
        target_dcm = dicom.dcmread(target_path)
        target_arr = target_dcm.pixel_array.astype(np.float32)
        
        # Normalize while preserving ranges
        input_norm, input_range = self.normalize(input_arr)
        target_norm, target_range = self.normalize(target_arr)
        
        # Convert to tensors
        input_tensor = torch.tensor(input_norm).unsqueeze(0)
        target_tensor = torch.tensor(target_norm).unsqueeze(0)
        
        if self.prediction_mode:
            return input_tensor, target_tensor, (input_range, target_range)
        return input_tensor, target_tensor

    def save_images(self, output_dir="generated_test/analyze"):
        os.makedirs(f"{output_dir}/input", exist_ok=True)
        os.makedirs(f"{output_dir}/target", exist_ok=True)
        
        for idx in tqdm(range(len(self))):
            input_path, target_path = self.folder[idx]
            input_tensor, target_tensor, (input_range, target_range) = self[idx]
            
            # Denormalize
            input_arr = self.denormalize(input_tensor.squeeze().numpy(), input_range)
            target_arr = self.denormalize(target_tensor.squeeze().numpy(), target_range)
            
            # Save DICOM
            for arr, orig_path, prefix in [(input_arr, input_path, 'input'), 
                                         (target_arr, target_path, 'target')]:
                orig_dcm = dicom.dcmread(orig_path)
                new_dcm = orig_dcm.copy()
                new_dcm.PixelData = arr.astype(orig_dcm.pixel_array.dtype).tobytes()
                new_dcm.SOPInstanceUID = dicom.uid.generate_uid()
                new_dcm.save_as(f"{output_dir}/{prefix}/{idx}.dcm")
    def __len__(self):
        return len(self.folder)
    
 
    
class Stacked3DDataset(Dataset):
    def __init__(self,
                 folder = "datav2/protocole_1/",
                 transform = transforms.Compose([
                    transforms.Resize((512, 512), antialias=True),
                    ])
            ):
        self.folder = load_all_acquisition(folder)
        self.transform = transform
        self.augmentation = None
        self.n_class = None

    def __len__(self):
        return len(self.folder) 
    
    def __getitem__(self, idx):
        all_acquisition = self.folder[idx]
        stacked_input = []
        stacked_target = []
        for image in all_acquisition:
            input_path, target_path = image
            input = np.array(dicom.dcmread(input_path).pixel_array, dtype=np.float32)
            target = np.array(dicom.dcmread(target_path).pixel_array, dtype=np.float32)
            input = normalize_ct_image(input, normalization_type='minmax')
            target = normalize_ct_image(target, normalization_type='minmax')
            input = torch.tensor(input).unsqueeze(0)
            target = torch.tensor(target).unsqueeze(0)
            if self.transform:
                input = self.transform(input)
                target = self.transform(target)
            if self.augmentation:
                input, target = self.augmentation(input, target)
            stacked_input.append(input)
            stacked_target.append(target)
        stacked_input = torch.stack(stacked_input)
        stacked_target = torch.stack(stacked_target)
        return stacked_input.permute(1,0,2,3), stacked_target.permute(1,0,2,3)




class LoadOneAcquisition(Dataset):
    def __init__(self,
                 path = "datav2/protocole_1/",
                 control = True,
                 categorie = "cocrhigh",
                 acquisition = 1,
                 augmentation = None,
                 generating = False
                 ) -> None:
        super().__init__()
        self.folder = load_one_acquisition(
            path = path,
            control = control,
            categorie = categorie,
            acquisition = acquisition
        )
        self.control = control
        self.augmentation = augmentation
        self.generating = generating
    
    def __len__(self):
        return len(self.folder)
    
    def normalize(self, image_array):
        """Simple min-max normalization"""
        min_val = image_array.min()
        max_val = image_array.max()
        return (image_array - min_val) / (max_val - min_val), (min_val, max_val)

    def denormalize(self, normalized_array, original_range):
        """Restore original values"""
        min_val, max_val = original_range
        return normalized_array * (max_val - min_val) + min_val

    def __getitem__(self, idx):
        input_path, target_path = self.folder[idx]
        
        # Load input
        input_dcm = dicom.dcmread(input_path)
        input_arr = input_dcm.pixel_array.astype(np.float32)
        
        # Load target
        target_dcm = dicom.dcmread(target_path)
        target_arr = target_dcm.pixel_array.astype(np.float32)
        
        # Normalize while preserving ranges
        input_norm, input_range = self.normalize(input_arr)
        target_norm, target_range = self.normalize(target_arr)
        
        # Convert to tensors
        input_tensor = torch.tensor(input_norm).unsqueeze(0)
        target_tensor = torch.tensor(target_norm).unsqueeze(0)
        
        return input_tensor, target_tensor, (input_range, target_range)


    def check_normalization(self):
        for idx in range(len(self)):
            input, target,*args = self[idx]
            print(f"Input range: {input.min():.3f} to {input.max():.3f}")
            print(f"Target range: {target.min():.3f} to {target.max():.3f}")
            # Should see values between 0 and 1
    
    def save_images(self, output_dir="generated_test/analyze"):
        os.makedirs(f"{output_dir}/input", exist_ok=True)
        os.makedirs(f"{output_dir}/target", exist_ok=True)
        
        for idx in tqdm(range(len(self))):
            input_path, target_path = self.folder[idx]
            input_tensor, target_tensor, (input_range, target_range) = self[idx]
            # Denormalize
            input_arr = self.denormalize(input_tensor.squeeze().numpy(), input_range)
            target_arr = self.denormalize(target_tensor.squeeze().numpy(), target_range)
            
            # Save DICOM
            for arr, orig_path, prefix in [(input_arr, input_path, 'input'), 
                                         (target_arr, target_path, 'target')]:
                orig_dcm = dicom.dcmread(orig_path)
                new_dcm = orig_dcm.copy()
                new_dcm.PixelData = arr.astype(orig_dcm.pixel_array.dtype).tobytes()
                new_dcm.SOPInstanceUID = dicom.uid.generate_uid()
                new_dcm.save_as(f"{output_dir}/{prefix}/{idx}.dcm")

    def generate(self, model, output_dir, device = "cuda"):
        os.makedirs(f"{output_dir}/input", exist_ok=True)
        os.makedirs(f"{output_dir}/target", exist_ok=True)
        os.makedirs(f"{output_dir}/generated", exist_ok=True)

        for idx in tqdm(range(len(self))):
            input_path, target_path = self.folder[idx]
            input_tensor, target_tensor, (input_range, target_range) = self[idx]
            generated_tensor = model(input_tensor.unsqueeze(0).to(device)).squeeze(0)

            
            
            input_arr = self.denormalize(input_tensor.squeeze().numpy(), input_range)
            target_arr = self.denormalize(target_tensor.squeeze().numpy(), target_range)
            generated_arr = self.denormalize(generated_tensor.squeeze().detach().cpu().numpy(), target_range)
            # Save DICOM
            for arr, orig_path, prefix in [(input_arr, input_path, 'input'), 
                                         (target_arr, target_path, 'target'),
                                         (generated_arr, target_path, 'generated')]:
                orig_dcm = dicom.dcmread(orig_path)
                new_dcm = orig_dcm.copy()
                new_dcm.PixelData = arr.astype(orig_dcm.pixel_array.dtype).tobytes()
                new_dcm.SOPInstanceUID = dicom.uid.generate_uid()
                new_dcm.save_as(f"{output_dir}/{prefix}/{idx}.dcm")


class Datav2Module(pl.LightningDataModule):
    def __init__(self,
                 folder = "datav2/protocole_1/",
                 dataset_type = Datav2Dataset,
                 train_bs = 1,
                 test_bs = 1,
                 train_ratio = 0.8,
                 data_folder = "complete",
                 img_size = 512,
                 pin_memory=True,
                 *args, **kwargs):
        
        self.folder = folder
        self.train_bs = train_bs
        self.test_bs = test_bs
        self.train_ratio = train_ratio
        self.valid_ratio = (1 - train_ratio)/2
        self.test_ratio = self.valid_ratio
            
        self.dataset_type = dataset_type
        self.data_folder = data_folder
        self.pin_memory = pin_memory
        self.img_size = img_size
        self.num_workers = self.get_optimal_num_workers()
        self.augmentation_dict = {
            ClassificationDataset :  DicomClassificationCollator(
                    prob_augment=0.5,  # 50% chance of applying an augmentation
                    rotation_range=(-10, 10),
                    scale_range=(0.95, 1.05),
                    brightness_range=(0.9, 1.1),
                    contrast_range=(0.9, 1.1),
                    noise_std=0.02,
                    enable_elastic=False  # Set to True if you want elastic deformation
            ),
            Datav2Dataset: DicomPredictionCollator(prob_augment=0.5,  # 50% chance of applying an augmentation
                    rotation_range=(-10, 10),
                    scale_range=(0.95, 1.05),
                    brightness_range=(0.9, 1.1),
                    contrast_range=(0.9, 1.1),
                    noise_std=0.02,
                    enable_elastic=False  # Set to True if you want elastic deformation
            )
        }
        self.args = args
        self.kwargs = kwargs

    def get_optimal_num_workers(self):
        slurms_cpu = os.environ.get('SLURM_CPUS_PER_TASK')
        if slurms_cpu is not None:
            num_cpus = int(slurms_cpu)
        else:
            num_cpus = os.cpu_count()
        if int(os.environ.get('LOCAL_RANK', 0)) == 0:
            print(f"Number of CPUs: {num_cpus}")    
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            cpu_used = min(num_cpus, 4 * num_gpus, 7)
        else:
            cpu_used = min(num_cpus, 8)  # Cap at 8 for CPU-only machines
        if int(os.environ.get('LOCAL_RANK', 0)) == 0:
            print(f"Number of workers: {cpu_used}")
        return cpu_used
        
    def setup(self, stage = None):
        self.dataset = self.dataset_type(folder = self.folder, data_folder=self.data_folder, img_size=self.img_size, *self.args, **self.kwargs)
        self.n_class = self.dataset.n_class
        total = len(self.dataset)
        train_size = int(self.train_ratio * total)
        valid_size = int(self.valid_ratio * total)
        test_size = total - train_size - valid_size
        self.train_ds, self.valid_ds, self.test_ds = torch.utils.data.random_split(self.dataset, [train_size, valid_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_ds, 
                          batch_size=self.train_bs, 
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          shuffle=True,
                          collate_fn=self.augmentation_dict[self.dataset_type])
    
    def val_dataloader(self):
        return DataLoader(self.valid_ds, 
                          batch_size=self.test_bs, 
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, 
                          batch_size=self.test_bs, 
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          shuffle=False)
    
    def combined_dataloader(self):
        return DataLoader(self.dataset, 
                          batch_size=self.train_bs, 
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          shuffle=False)




if __name__ == "__main__":
    """ ds = Datav2Dataset("/media/gabrielidis/LaCie/Hugo/dataset/medicalv2/protocole_1/", prediction_mode=True)
    ds.save_images()
    """   
    datav2 = Datav2Module(folder="/media/gabrielidis/LaCie/Hugo/dataset/medicalv2/protocole_1/")
    acquisition_number = 1
    categorie = "guttalow"
    ds = LoadOneAcquisition(
        control=False,
        path = "/media/gabrielidis/LaCie/Hugo/dataset/medicalv2/protocole_1/",
        categorie = categorie,
        acquisition = acquisition_number,
    )
    #ds.check_normalization()
    ds.save_images()