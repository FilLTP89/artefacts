import random
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF

def normalize_ct_image(dicom_image, normalization_type='minmax', custom_range=None):
    # Apply rescale
    hu_image = dicom_image * 1 - 1000  # Using the rescale values from your results

    if normalization_type == 'full_range':
        # Normalize to full HU range
        normalized = (hu_image - (-1024)) / (3071 - (-1024))
    elif normalization_type == 'window':
        # Window-level normalization (example for soft tissue window)
        window_center = 40
        window_width = 400
        window_min = window_center - window_width // 2
        window_max = window_center + window_width // 2
        normalized = np.clip(hu_image, window_min, window_max)
        normalized = (normalized - window_min) / (window_max - window_min)
    elif normalization_type == 'minmax':
        # Min-max normalization to [0, 1]
        min_val = np.min(hu_image)
        max_val = np.max(hu_image)
        normalized = (hu_image - min_val) / (max_val - min_val)
    elif normalization_type == 'zscore':
        # Z-score normalization
        mean = np.mean(hu_image)
        std = np.std(hu_image)
        normalized = (hu_image - mean) / std
    elif normalization_type == 'custom':
        # Custom range normalization
        if custom_range is None:
            raise ValueError("Custom range must be provided for custom normalization")
        min_val, max_val = custom_range
        normalized = np.clip(hu_image, min_val, max_val)
        normalized = (normalized - min_val) / (max_val - min_val)
    elif normalization_type == 'simple':
        normalized = hu_image / np.max(hu_image)
    else:
        raise ValueError("Invalid normalization type")

    return normalized

class CTImageAugmentation(nn.Module):
    """
    Random Rotation: Small rotations to simulate patient positioning 
        variations.
    Random Zoom: Slight zooming in/out to simulate variations in 
        scan parameters or patient size.
    Random Flip: Horizontal flipping (be cautious with this for 
        certain anatomical structures).
    Random Shift: Small translations to simulate patient movement
        or positioning differences.
    Random Noise: Adding noise to simulate variations in image 
        quality or scanner characteristics.
    Random Contrast: Adjusting contrast to simulate variations 
        in tissue density or scanner settings.
    Random Gamma: Altering gamma to simulate variations in 
        image brightness and contrast.
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def random_rotate(self, image, max_angle=10):
        if random.random() < self.p:
            angle = random.uniform(-max_angle, max_angle)
            return TF.rotate(image, angle, interpolation=TF.InterpolationMode.NEAREST)
        return image

    def random_zoom(self, image, zoom_range=(0.9, 1.1)):
        if random.random() < self.p:
            zoom_factor = random.uniform(zoom_range[0], zoom_range[1])
            h, w = image.shape[-2:]
            new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
            return TF.resize(image, (new_h, new_w), interpolation=TF.InterpolationMode.NEAREST)
        return image

    def random_flip(self, image):
        if random.random() < self.p:
            return TF.hflip(image)
        return image

    def random_shift(self, image, max_shift=10):
        if random.random() < self.p:
            shift_x = random.randint(-max_shift, max_shift)
            shift_y = random.randint(-max_shift, max_shift)
            return TF.affine(image, angle=0, translate=(shift_x, shift_y), scale=1, shear=0)
        return image

    def random_noise(self, image, noise_std=0.01):
        if random.random() < self.p:
            noise = torch.randn_like(image) * noise_std
            return image + noise
        return image

    def random_contrast(self, image, factor_range=(0.8, 1.2)):
        if random.random() < self.p:
            factor = random.uniform(factor_range[0], factor_range[1])
            mean = image.mean()
            return (image - mean) * factor + mean
        return image

    def random_gamma(self, image, gamma_range=(0.8, 1.2)):
        if random.random() < self.p:
            gamma = random.uniform(gamma_range[0], gamma_range[1])
            return image.pow(gamma)
        return image

    def forward(self, image):
        # Ensure image is a 4D tensor (batch, channel, height, width)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Apply augmentations
        image = self.random_rotate(image)
        image = self.random_zoom(image)
        image = self.random_flip(image)
        image = self.random_shift(image)
        image = self.random_noise(image)
        image = self.random_contrast(image)
        image = self.random_gamma(image)
        
        # Remove batch dimension if it was added
        if image.size(0) == 1:
            image = image.squeeze(0)
        
        return image
