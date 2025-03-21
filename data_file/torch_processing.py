import os
#from data_file.visualize import visualize_from_dataset    
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
from sklearn.model_selection import train_test_split
#import h5py
from PIL import Image
import torchvision.transforms as transforms
import pytorch_lightning as pl
from data_file.CBCT_preprocess import read_raw




class PyTorchDataset(Dataset):
    def __init__(
        self,
        path: str = "./data/",
        width: int = 512,
        height: int = 512,
        saving_format: str = None,
        train_saving_path: str = "train/",
        test_saving_path: str = "test/",
        valid_saving_path: str = "valid/",
        seed: int = 42,
        big_endian: bool = True,
        shuffle: bool = False,
        mode: str = "train"  # 'train', 'valid', or 'test'
    ) -> None:
        """
        PyTorch Dataset for metal artifact reduction
        
        Args:
            path (str): Path to the data directory
            width (int): Target width for resizing
            height (int): Target height for resizing
            saving_format (str): Format to save the dataset
            train_saving_path (str): Path to save the training dataset
            test_saving_path (str): Path to save the testing dataset
            valid_saving_path (str): Path to save the validation dataset
            seed (int): Random seed for reproducibility
            big_endian (bool): Whether the raw files are in big endian format
            shuffle (bool): Whether to shuffle the dataset
            mode (str): Mode of the dataset ('train', 'valid', or 'test')
        """
        self.path = path
        self.width = width
        self.height = height
        self.saving_format = saving_format
        self.train_saving_path = path + train_saving_path
        self.test_saving_path = path + test_saving_path
        self.valid_saving_path = path + valid_saving_path
        
        self.original_width = 400
        self.original_height = 400
        
        self.seed = seed
        self.big_endian = big_endian
        self.shuffle = shuffle
        self.mode = mode
        
        # Initialize the data
        self.collect_data()
        self.setup()
        
        # Define transforms for resizing
        self.transform = transforms.Compose([
            transforms.Resize((self.height, self.width))
        ])
    
    def collect_data(self):
        """
        Create lists of paths of the raw images from each folder
        """
        # Collect paths for No_metal folder
        no_metal_folder = [
            sorted(
                glob(os.path.join(self.path, "No_metal/acquisition_" + str(i) + "/*")),
                key=lambda x: [
                    int(c) if c.isdigit() else c for c in re.split(r"(\d+)", x)
                ],
            )
            for i in range(11)
        ]
        
        # Collect paths for High_metal folder
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
        
        # Collect paths for Low_metal folder
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
        
        # Flatten the lists
        no_metal_list = [item for sublist in no_metal_folder for item in sublist]
        high_metal_list = [item for sublist in high_metal_folder for item in sublist]
        low_metal_list = [item for sublist in low_metal_folder for item in sublist]
        
        self.no_metal_list = no_metal_list
        self.high_metal_list = high_metal_list
        self.low_metal_list = low_metal_list
    
    def setup(self):
        """
        Split the data into training, validation, and testing sets
        """
        # Create input-label pairs
        label = 2 * self.no_metal_list  # Duplicating to match the input size
        input_list = self.high_metal_list + self.low_metal_list
        
        # Split the data
        X_train, X_test_1, y_train, y_test_1 = train_test_split(
            input_list, label, test_size=(2 / 11), random_state=self.seed, shuffle=self.shuffle
        )  # 8 acquisition for training
        
        X_test, X_valid, y_test, y_valid = train_test_split(
            X_test_1, y_test_1, test_size=0.5, random_state=self.seed, shuffle=self.shuffle
        )  # 1 acquisition for validation & 1 acquisition for testing
        
        self.X_train, self.y_train = X_train, y_train
        self.X_valid, self.y_valid = X_valid, y_valid
        self.X_test, self.y_test = X_test, y_test
        
        # Set the active dataset based on the mode
        if self.mode == "train":
            self.X, self.y = self.X_train, self.y_train
        elif self.mode == "valid":
            self.X, self.y = self.X_valid, self.y_valid
        elif self.mode == "test":
            self.X, self.y = self.X_test, self.y_test
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def __len__(self):
        """
        Return the length of the dataset
        """
        return len(self.X)
    
    def preprocess(self, x_path, y_path):
        """
        Process the raw files into normalized tensors
        
        Args:
            x_path (str): Path to the input raw file
            y_path (str): Path to the label raw file
            
        Returns:
            tuple: Processed input and label tensors
        """
        # Read the raw files
        with_artifact = read_raw(
            x_path,
            image_size=(self.original_height, self.original_width),
            big_endian=self.big_endian,
        )
        
        without_artifact = read_raw(
            y_path,
            image_size=(self.original_height, self.original_width),
            big_endian=self.big_endian,
        )
        
        # Convert to PyTorch tensors
        input_tensor = torch.from_numpy(with_artifact).float().unsqueeze(0)  # Add channel dimension
        label_tensor = torch.from_numpy(without_artifact).float().unsqueeze(0)  # Add channel dimension
        
        # Resize if needed
        if self.width != self.original_width or self.height != self.original_height:
            input_tensor = transforms.functional.resize(input_tensor, (self.height, self.width))
            label_tensor = transforms.functional.resize(label_tensor, (self.height, self.width))
            
        return input_tensor, label_tensor
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset by index
        
        Args:
            idx (int): Index
            
        Returns:
            tuple: Input and label tensors
        """
        x_path = self.X[idx]
        y_path = self.y[idx]
        
        return self.preprocess(x_path, y_path)
    
    """ def save(self):
        #Save the dataset to disk
        if self.saving_format in ["hdf5", "h5"]:
            # Save as HDF5
            os.makedirs(os.path.join(self.path, "save"), exist_ok=True)
            
            # Save train dataset
            with h5py.File(f"{self.path}/save/train_dataset.h5", "w") as f:
                x_dset = f.create_dataset("inputs", (len(self.X_train), 1, self.height, self.width), dtype='f')
                y_dset = f.create_dataset("labels", (len(self.y_train), 1, self.height, self.width), dtype='f')
                
                for i in range(len(self.X_train)):
                    x, y = self.preprocess(self.X_train[i], self.y_train[i])
                    x_dset[i] = x.numpy()
                    y_dset[i] = y.numpy()
            
            # Save validation dataset
            with h5py.File(f"{self.path}/save/valid_dataset.h5", "w") as f:
                x_dset = f.create_dataset("inputs", (len(self.X_valid), 1, self.height, self.width), dtype='f')
                y_dset = f.create_dataset("labels", (len(self.y_valid), 1, self.height, self.width), dtype='f')
                
                for i in range(len(self.X_valid)):
                    x, y = self.preprocess(self.X_valid[i], self.y_valid[i])
                    x_dset[i] = x.numpy()
                    y_dset[i] = y.numpy()
            
            # Save test dataset
            with h5py.File(f"{self.path}/save/test_dataset.h5", "w") as f:
                x_dset = f.create_dataset("inputs", (len(self.X_test), 1, self.height, self.width), dtype='f')
                y_dset = f.create_dataset("labels", (len(self.y_test), 1, self.height, self.width), dtype='f')
                
                for i in range(len(self.X_test)):
                    x, y = self.preprocess(self.X_test[i], self.y_test[i])
                    x_dset[i] = x.numpy()
                    y_dset[i] = y.numpy()
        else:
            # Save paths
            os.makedirs(self.train_saving_path, exist_ok=True)
            os.makedirs(self.valid_saving_path, exist_ok=True)
            os.makedirs(self.test_saving_path, exist_ok=True)
            
            # Save the paths to text files
            with open(os.path.join(self.train_saving_path, "inputs.txt"), "w") as f:
                f.write("\n".join(self.X_train))
            with open(os.path.join(self.train_saving_path, "labels.txt"), "w") as f:
                f.write("\n".join(self.y_train))
                
            with open(os.path.join(self.valid_saving_path, "inputs.txt"), "w") as f:
                f.write("\n".join(self.X_valid))
            with open(os.path.join(self.valid_saving_path, "labels.txt"), "w") as f:
                f.write("\n".join(self.y_valid))
                
            with open(os.path.join(self.test_saving_path, "inputs.txt"), "w") as f:
                f.write("\n".join(self.X_test))
            with open(os.path.join(self.test_saving_path, "labels.txt"), "w") as f:
                f.write("\n".join(self.y_test))
     """
    """ 
    def load(self):
        #Load the dataset from disk
        if self.saving_format in ["hdf5", "h5"]:
            # Load is handled in __getitem__ for HDF5 format
            # This method would update the internal state to use the saved files instead
            self.use_saved = True
        else:
            # Load paths from text files
            with open(os.path.join(self.train_saving_path, "inputs.txt"), "r") as f:
                self.X_train = f.read().splitlines()
            with open(os.path.join(self.train_saving_path, "labels.txt"), "r") as f:
                self.y_train = f.read().splitlines()
                
            with open(os.path.join(self.valid_saving_path, "inputs.txt"), "r") as f:
                self.X_valid = f.read().splitlines()
            with open(os.path.join(self.valid_saving_path, "labels.txt"), "r") as f:
                self.y_valid = f.read().splitlines()
                
            with open(os.path.join(self.test_saving_path, "inputs.txt"), "r") as f:
                self.X_test = f.read().splitlines()
            with open(os.path.join(self.test_saving_path, "labels.txt"), "r") as f:
                self.y_test = f.read().splitlines()
            
            # Update the current data based on mode
            if self.mode == "train":
                self.X, self.y = self.X_train, self.y_train
            elif self.mode == "valid":
                self.X, self.y = self.X_valid, self.y_valid
            elif self.mode == "test":
                self.X, self.y = self.X_test, self.y_test
    """
    def load_single_acquisition(self, acquisition_number=1, low=False):
        """
        Create a dataset for a single acquisition
        
        Args:
            acquisition_number (int): The acquisition number to load
            low (bool): Whether to use low metal data (True) or high metal data (False)
            
        Returns:
            PyTorchDataset: A new dataset for the specified acquisition
        """
        # Create a new dataset with the same parameters
        dataset = PyTorchDataset(
            path=self.path,
            width=self.width,
            height=self.height,
            saving_format=self.saving_format,
            train_saving_path=self.train_saving_path,
            test_saving_path=self.test_saving_path,
            valid_saving_path=self.valid_saving_path,
            seed=self.seed,
            big_endian=self.big_endian,
            shuffle=self.shuffle,
            mode="train"  # Mode doesn't matter here as we'll override the data
        )
        
        # Set the data to the specified acquisition
        no_metal_folder = self.no_metal_folder[acquisition_number]
        metal_folder = self.low_metal_folder[acquisition_number] if low else self.high_metal_folder[acquisition_number]
        
        dataset.X = metal_folder
        dataset.y = no_metal_folder
        
        return dataset



class MetalArtifactDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = "data/",
        width: int = 512,
        height: int = 512,
        train_bs: int = 32,
        test_bs = 32,
        num_workers: int = 4,
        big_endian: bool = True,
        shuffle: bool = True,
        seed: int = 42,
        *args,
        **kwargs
    ):
        """
        PyTorch Lightning DataModule for the Metal Artifact Reduction dataset
        
        Args:
            data_path (str): Path to the data directory
            width (int): Target width for images
            height (int): Target height for images
            batch_size (int): Batch size for dataloaders
            num_workers (int): Number of workers for dataloaders
            big_endian (bool): Whether raw files are in big endian format
            shuffle (bool): Whether to shuffle the training data
            seed (int): Random seed for reproducibility
        """
        super().__init__()
        self.data_path = data_path
        self.width = width
        self.height = height
        self.train_bs = train_bs
        self.test_bs = test_bs
        self.num_workers = num_workers
        self.big_endian = big_endian
        self.shuffle = shuffle
        self.seed = seed
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def prepare_data(self):
        """
        Prepare data method
        This method is called only once and on 1 GPU
        Use this to download/prepare data (one-time operations)
        """
        # We don't need to download anything since data is already available
        # But we could add any one-time operations like checking if files exist
        pass
    
    def setup(self, stage = None):
        """
        Setup datasets for trainer stages (fit, validate, test)
        This method is called on every process when using DDP
        
        Args:
            stage (str, optional): Current stage ('fit', 'validate', 'test', or None)
        """
        # Create datasets for the specified stage
        if stage == 'fit' or stage is None:
            self.train_dataset = PyTorchDataset(
                path=self.data_path,
                width=self.width,
                height=self.height,
                big_endian=self.big_endian,
                shuffle=self.shuffle,
                seed=self.seed,
                mode="train"
            )
            
            self.val_dataset = PyTorchDataset(
                path=self.data_path,
                width=self.width,
                height=self.height,
                big_endian=self.big_endian,
                shuffle=False,  # No need to shuffle validation data
                seed=self.seed,
                mode="valid"
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = PyTorchDataset(
                path=self.data_path,
                width=self.width,
                height=self.height,
                big_endian=self.big_endian,
                shuffle=False,  # No need to shuffle test data
                seed=self.seed,
                mode="test"
            )
    
    def train_dataloader(self):
        """
        Create the training dataloader
        
        Returns:
            DataLoader: Training data loader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_bs,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        """
        Create the validation dataloader
        
        Returns:
            DataLoader: Validation data loader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.test_bs,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        """
        Create the test dataloader
        
        Returns:
            DataLoader: Test data loader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_bs,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_single_acquisition_dataloader(self, acquisition_number=1, low=False, batch_size=None):
        """
        Create a dataloader for a single acquisition
        
        Args:
            acquisition_number (int): The acquisition number to load
            low (bool): Whether to use low metal data (True) or high metal data (False)
            batch_size (int, optional): Batch size override for this dataloader
            
        Returns:
            DataLoader: DataLoader for the specified acquisition
        """
        # Use train_dataset to get acquisition (it has all folders loaded)
        if self.train_dataset is None:
            self.setup()
        
        single_acquisition_dataset = self.train_dataset.load_single_acquisition(
            acquisition_number=acquisition_number,
            low=low
        )
        
        return DataLoader(
            single_acquisition_dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

# Example usage
if __name__ == "__main__":
    # Create dataset
    print("Using pytorch dataset")
    dataset = PyTorchDataset(path="/media/gabrielidis/LaCie/Hugo/dataset/medical/", width=512, height=512, big_endian=True, shuffle=True, mode="train")
    dataset.setup()
    x,y = dataset[0]
    print(x.shape, y.shape)
    

