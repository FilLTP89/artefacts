import torch
import numpy as np
from torch.utils.data import Dataset
from datasets import Dataset as DS
import cv2
from glob import glob
import os
from PIL import Image
import random
import tifffile
import matplotlib.pyplot as plt
from transformers import SamProcessor
from scipy.ndimage import label
from torch.utils.data import DataLoader
import pickle

np.random.seed = 42

def plot_mask_with_bboxes(img,ground_truth_map, bounding_boxes):
    fig, ax = plt.subplots(1,2,figsize=(20, 20))
    ax[0].imshow(img, cmap = "gray")
    ax[1].imshow(ground_truth_map, cmap='gray')

    for bbox in bounding_boxes:
        x_min, y_min, x_max, y_max = bbox
        ax[1].add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, edgecolor='red', linewidth=2))

    plt.show()


def load_image_and_mask(path = "./data/segmentation"):
    images_path = f"{path}/images/"
    masks_path = f"{path}/masks/"

    images_folder = [
            sorted(
                glob(os.path.join(images_path, str(i) + "/*.tif")),
            )
            for i in range(2,6)
        ]
    mask_folder = [
            sorted(
                glob(os.path.join(masks_path, str(i) + "/*.tif")),
            )
            for i in range(2,6)
        ]

    images_list = [item for sublist in images_folder for item in sublist]
    masks_list = [item for sublist in mask_folder for item in sublist]
    dataset_dict = {
    "image":  images_list,
    "label":  masks_list,
    }
    return DS.from_dict(dataset_dict)

def has_more_zeros(binary_image):
    num_ones = np.sum(binary_image)
    total_pixels = binary_image.size

    return num_ones < total_pixels / 2



def get_bounding_boxes(ground_truth_map):
    # Find connected components
    labeled, num_features = label(ground_truth_map > 0)

    bounding_boxes = []

    for feature in range(1, num_features + 1):
        y_indices, x_indices = np.where(labeled == feature)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # add perturbation to bounding box coordinates
        H, W = ground_truth_map.shape
        x_min = max(0, x_min)
        x_max = min(W, x_max)
        y_min = max(0, y_min)
        y_max = min(H, y_max)
        bbox = [x_min, y_min, x_max, y_max]

        bounding_boxes.append(np.array(bbox))

    return np.array(bounding_boxes)



class SAMDataset(Dataset):
    def __init__(self, dataset = "data/segmentation" , processor = "facebook/sam-vit-base", precomputed_data_path="data/segmentation/precomputed_data.pkl"):
        self.processor = SamProcessor.from_pretrained(processor)
        self.dataset = load_image_and_mask(dataset)
        if precomputed_data_path and os.path.exists(precomputed_data_path):
            self.load_precomputed_data(precomputed_data_path)
        else:
            self.saved_data = []
            self.precompute_data(self.dataset)
            if precomputed_data_path:
                self.save_precomputed_data(precomputed_data_path)

    def precompute_data(self, dataset):
        for item in dataset:
            ground_truth_mask = np.array(tifffile.imread(item["label"])[:, :, 0])
            ground_truth_mask = ground_truth_mask / 255.0
            ground_truth_mask = np.where(ground_truth_mask > 0.5, 1, 0)
            bboxes = get_bounding_boxes(ground_truth_mask)
            for bbox in bboxes:
                # Store only the file paths and bounding box coordinates
                self.saved_data.append((item["image"], item["label"], bbox.tolist()))

    def save_precomputed_data(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self.saved_data, file)

    def load_precomputed_data(self, file_path):
        with open(file_path, 'rb') as file:
            self.saved_data = pickle.load(file)
    
    def __len__(self):
        return len(self.saved_data)
    
    def __getitem__(self, idx):
        if isinstance(idx, list):
            # This is unexpected; we should only receive individual indices
            raise ValueError("Received a list of indices, expected a single index")
        
        image_path, label_path, bbox = self.saved_data[idx]
        image = np.array(tifffile.imread(image_path))
        ground_truth_mask = np.array(tifffile.imread(label_path)[:, :, 0])
        ground_truth_mask = ground_truth_mask / 255.0
        ground_truth_mask = np.where(ground_truth_mask > 0.5, 1, 0)
        ground_truth_mask = ground_truth_mask[np.newaxis, :]
        prompt = [[float(num) for num in bbox]]
        inputs = self.processor(image, input_boxes=[prompt], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = torch.tensor(ground_truth_mask)
        return inputs

class SAMModule:
    def __init__(self, path = "data/segmentation",
                 precached_data_path = "data/segmentation/precomputed_data.pkl", 
                 model_name="facebook/sam-vit-base",
                batch_size = 3 ) -> None:

        ds = SAMDataset(precomputed_data_path=precached_data_path, processor=model_name)
        self.train_loader = DataLoader(ds, 
                                       batch_size=batch_size,
                                         shuffle=False,num_workers=8, collate_fn= self.collate_fn)
    
    def collate_fn(self,batch):
    # Initialize a dictionary to hold the collated batch
        collated_batch = {}

        # Iterate over all keys in the dictionary returned by __getitem__
        for key in batch[0]:
            # Collate each tensor in the list using torch.stack or similar method
            collated_batch[key] = torch.stack([item[key] for item in batch])
        return collated_batch
    

if __name__ == "__main__":
    print("Generating dataset...")
    dataset = load_image_and_mask("../data/segmentation")
    processor = "facebook/sam-vit-base"
    print("Dataset generated!")
    n = 0
    for _ in range(n):
        img_num = random.randint(0, len(dataset)-1)
        example_image = dataset[img_num]["image"]
        example_mask = dataset[img_num]["label"]
        img = tifffile.imread(example_image)[:,:,0]
        mask = tifffile.imread(example_mask)
        mask_t = mask[:,:,0] / 255.0
        mask_t = np.where(mask_t > 0.5, 1, 0)
        bboxs = get_bounding_boxes(mask_t)
        bboxs = [[float(item) for item in sublist] for sublist in bboxs]
        if len(bboxs) >2:
            print(len(bboxs))
            plot_mask_with_bboxes(img,mask_t, bboxs)    
    from transformers import SamModel
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    module = SAMModule(precached_data_path="../data/segmentation/precomputed_data.pkl", batch_size= 1)
    train_loader = module.train_loader
    for idx,batch in enumerate(train_loader):
        outputs = model(**batch)
        print(outputs.keys())
        break

    """
    Issue :  there is a differents number of boxes for each images
    Hugging face answer : 
    If you want to fine-tune SAM with multiple bounding boxes, 
    you need to create several training examples, each containing a single 
    (image, box, mask) triplet.
    """

    """
    outputs = model(**ds[10])
    """