import torch
import numpy as np
from datasets import Dataset
import cv2
from glob import glob
import os
from PIL import Image
import random
import tifffile
import matplotlib.pyplot as plt
from transformers import SamProcessor
from scipy.ndimage import label


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
    return Dataset.from_dict(dataset_dict)
def has_more_zeros(binary_image):
    num_ones = np.sum(binary_image)
    total_pixels = binary_image.size

    return num_ones < total_pixels / 2

class SAMDataset(Dataset):
  """
  This class is used to create a dataset that serves input images and masks.
  It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
  """
  def __init__(self, dataset, processor):
    self.dataset = dataset
    self.processor = processor

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    image = np.array(tifffile.imread(item["image"])) 
    ground_truth_mask = np.array(tifffile.imread(item["label"])[:,:,0])
    
    # convert to binary mask
    ground_truth_mask = ground_truth_mask / 255.0
    ground_truth_mask = np.where(ground_truth_mask > 0.5, 1, 0)
       
    # get bounding box prompt
    prompt = get_bounding_boxes(ground_truth_mask)
    prompt = [[float(item) for item in sublist] for sublist in prompt]
    # add batch dimension
    before_processing_image = image.copy()
    before_processing_mask = mask.copy()
    ground_truth_mask  = ground_truth_mask[np.newaxis,:]
    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=[prompt], return_tensors="pt")
    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}
    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask
    inputs["original_image"] = before_processing_image
    inputs["original_mask"] = before_processing_mask
    return inputs
  

if __name__ == "__main__":
    print("Generating dataset...")
    dataset = load_image_and_mask("../data/segmentation")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    print("Dataset generated!")
    n = 5
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
        #plot_mask_with_bboxes(img,mask_t, bboxs)    
    ds = SAMDataset(dataset, processor) 
    print(ds[0].keys())
    print(ds[10]["original_mask"].shape)
    print(ds[10]["original_image"].shape)