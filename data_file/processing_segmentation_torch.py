import os 
import torch
from glob import glob
import tifffile as tiff
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datasets import Dataset as DS
import torchvision.transforms as T


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

def show_image_and_mask(image, mask):
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(image[0],cmap="gray")
    ax[1].imshow(mask[0],cmap="gray")
    plt.show()


class SegmentationTorchDataset(Dataset):
    def __init__(self, dataset_path = "data/segmentation"):
        self.dataset = load_image_and_mask(dataset_path)
        self.minshape_resize = T.Resize((1024,1024))


    def __len__(self):
        return len(self.dataset)
    
    def to_mask(self, mask):
        mask = mask / 255.0
        mask = torch.where(mask > 0.5, 1, 0)
        return mask
    
    def to_image(self, image):
        image = image / 255.0
        return image
    
    def __getitem__(self, idx):
        # Load image and mask
        img_path = self.dataset[idx]["image"]
        mask_path = self.dataset[idx]["label"]
        image = torch.Tensor(tiff.imread(img_path))[:,:,0]
        ground_truth_mask = torch.Tensor(tiff.imread(mask_path)[:, :, 0])

        ground_truth_mask = self.to_mask(ground_truth_mask)
        image = self.to_image(image)

        image = self.minshape_resize(image[None, None, ...])
        ground_truth_mask = self.minshape_resize(ground_truth_mask[None, None, ...])
        return image[0,...], ground_truth_mask[0,...] # format is (1,1024,1024)
    

if __name__ == "__main__":
    ds = SegmentationTorchDataset(dataset_path="../data/segmentation")
    dataloader = DataLoader(ds, batch_size=16, shuffle=True)
    for i, (images, masks) in enumerate(dataloader):
        print(images.shape)
        print(masks.shape)
        show_image_and_mask(images[0], masks[0])
        if i == 5:
            break    