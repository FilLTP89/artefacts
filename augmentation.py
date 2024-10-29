import torch
import numpy as np
from typing import List, Tuple, Union
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data.dataloader import default_collate
import random
from enum import Enum

class AugmentationType(Enum):
    ROTATION = "rotation"
    SCALE = "scale"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    NOISE = "noise"
    ELASTIC = "elastic"

class DicomClassificationCollator:
    """
    A collate function that applies a single random augmentation to each image.
    """
    def __init__(
        self,
        prob_augment: float = 0.5,
        rotation_range: Tuple[float, float] = (-10, 10),
        scale_range: Tuple[float, float] = (0.95, 1.05),
        brightness_range: Tuple[float, float] = (0.9, 1.1),
        contrast_range: Tuple[float, float] = (0.9, 1.1),
        noise_std: float = 0.02,
        enable_elastic: bool = False,
        augmentation_weights: dict[AugmentationType, float] = None
    ):
        """
        Initialize the collator with augmentation parameters.
        
        Args:
            prob_augment: Probability of applying any augmentation
            rotation_range: Range of rotation angles in degrees
            scale_range: Range of scaling factors
            brightness_range: Range of brightness adjustment
            contrast_range: Range of contrast adjustment
            noise_std: Standard deviation for Gaussian noise
            enable_elastic: Whether to enable elastic deformation
            augmentation_weights: Dictionary of weights for each augmentation type
        """
        self.prob_augment = prob_augment
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std
        self.enable_elastic = enable_elastic
        
        # Default weights if none provided
        self.augmentation_weights = augmentation_weights or {
            AugmentationType.ROTATION: 1.0,
            AugmentationType.SCALE: 1.0,
            AugmentationType.BRIGHTNESS: 1.0,
            AugmentationType.CONTRAST: 1.0,
            AugmentationType.NOISE: 1.0,
            AugmentationType.ELASTIC: 1.0 if enable_elastic else 0.0
        }
        
        # Normalize weights
        total_weight = sum(self.augmentation_weights.values())
        self.augmentation_weights = {
            k: v/total_weight for k, v in self.augmentation_weights.items()
        }

    def _apply_rotation(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random rotation."""
        angle = random.uniform(*self.rotation_range)
        return TF.rotate(image, angle)

    def _apply_scale(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random scaling."""
        scale = random.uniform(*self.scale_range)
        return TF.affine(image, angle=0, translate=[0, 0], scale=scale, shear=[0, 0])

    def _apply_brightness(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random brightness adjustment."""
        brightness = random.uniform(*self.brightness_range)
        return TF.adjust_brightness(image, brightness)

    def _apply_contrast(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random contrast adjustment."""
        contrast = random.uniform(*self.contrast_range)
        return TF.adjust_contrast(image, contrast)

    def _apply_gaussian_noise(self, image: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian noise."""
        noise = torch.randn_like(image) * self.noise_std
        return torch.clamp(image + noise, 0, 1)

    def _apply_elastic_transform(
        self,
        image: torch.Tensor,
        alpha: float = 50,
        sigma: float = 5
    ) -> torch.Tensor:
        """Apply elastic deformation."""
        shape = image.shape[-2:]
        dx = torch.randn(shape) * alpha
        dy = torch.randn(shape) * alpha
        
        dx = TF.gaussian_blur(dx.unsqueeze(0), kernel_size=11, sigma=sigma).squeeze(0)
        dy = TF.gaussian_blur(dy.unsqueeze(0), kernel_size=11, sigma=sigma).squeeze(0)
        
        x, y = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]))
        indices = torch.stack([y + dy, x + dx])
        
        return TF.grid_sample(
            image.unsqueeze(0),
            indices.permute(1, 2, 0).unsqueeze(0),
            mode='bilinear',
            padding_mode='border'
        ).squeeze(0)

    def _augment_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """Apply a single random augmentation to the image."""
        if random.random() > self.prob_augment:
            return image, "none"

        # Choose random augmentation based on weights
        aug_type = random.choices(
            list(self.augmentation_weights.keys()),
            weights=list(self.augmentation_weights.values()),
            k=1
        )[0]

        # Apply chosen augmentation
        if aug_type == AugmentationType.ROTATION:
            return self._apply_rotation(image), "rotation"
        elif aug_type == AugmentationType.SCALE:
            return self._apply_scale(image), "scale"
        elif aug_type == AugmentationType.BRIGHTNESS:
            return self._apply_brightness(image), "brightness"
        elif aug_type == AugmentationType.CONTRAST:
            return self._apply_contrast(image), "contrast"
        elif aug_type == AugmentationType.NOISE:
            return self._apply_gaussian_noise(image), "noise"
        elif aug_type == AugmentationType.ELASTIC and self.enable_elastic:
            return self._apply_elastic_transform(image), "elastic"
        
        return image, "none"

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collate function to be used with DataLoader.
        
        Args:
            batch: List of tuples (image, target)
        
        Returns:
            Tuple containing:
                - Batch of augmented images: torch.Tensor of shape [B, 1, H, W]
                - Batch of targets: torch.LongTensor of shape [B]
        """
        images, targets = zip(*batch)
        
        # Apply single random augmentation to each image
        augmented_images, aug_types = zip(*[self._augment_image(image) for image in images])
        
        # For debugging/logging purposes, you can print augmentation types
        # print(f"Batch augmentations applied: {aug_types}")
        
        return torch.stack(augmented_images), torch.stack(targets)
    


class DicomPredictionCollator:
    """
    A collate function that applies a single random augmentation to each image.
    """
    def __init__(
        self,
        prob_augment: float = 0.5,
        rotation_range: Tuple[float, float] = (-10, 10),
        scale_range: Tuple[float, float] = (0.95, 1.05),
        brightness_range: Tuple[float, float] = (0.9, 1.1),
        contrast_range: Tuple[float, float] = (0.9, 1.1),
        noise_std: float = 0.02,
        enable_elastic: bool = False,
        augmentation_weights: dict[AugmentationType, float] = None
    ):
        """
        Initialize the collator with augmentation parameters.
        
        Args:
            prob_augment: Probability of applying any augmentation
            rotation_range: Range of rotation angles in degrees
            scale_range: Range of scaling factors
            brightness_range: Range of brightness adjustment
            contrast_range: Range of contrast adjustment
            noise_std: Standard deviation for Gaussian noise
            enable_elastic: Whether to enable elastic deformation
            augmentation_weights: Dictionary of weights for each augmentation type
        """
        self.prob_augment = prob_augment
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std
        self.enable_elastic = enable_elastic
        
        # Default weights if none provided
        self.augmentation_weights = augmentation_weights or {
            AugmentationType.ROTATION: 1.0,
            AugmentationType.SCALE: 1.0,
            AugmentationType.BRIGHTNESS: 1.0,
            AugmentationType.CONTRAST: 1.0,
            AugmentationType.NOISE: 1.0,
            AugmentationType.ELASTIC: 1.0 if enable_elastic else 0.0
        }
        
        # Normalize weights
        total_weight = sum(self.augmentation_weights.values())
        self.augmentation_weights = {
            k: v/total_weight for k, v in self.augmentation_weights.items()
        }

    def _apply_rotation(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random rotation."""
        angle = random.uniform(*self.rotation_range)
        return TF.rotate(image, angle)

    def _apply_scale(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random scaling."""
        scale = random.uniform(*self.scale_range)
        return TF.affine(image, angle=0, translate=[0, 0], scale=scale, shear=[0, 0])

    def _apply_brightness(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random brightness adjustment."""
        brightness = random.uniform(*self.brightness_range)
        return TF.adjust_brightness(image, brightness)

    def _apply_contrast(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random contrast adjustment."""
        contrast = random.uniform(*self.contrast_range)
        return TF.adjust_contrast(image, contrast)

    def _apply_gaussian_noise(self, image: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian noise."""
        noise = torch.randn_like(image) * self.noise_std
        return torch.clamp(image + noise, 0, 1)

    def _apply_elastic_transform(
        self,
        image: torch.Tensor,
        alpha: float = 50,
        sigma: float = 5
    ) -> torch.Tensor:
        """Apply elastic deformation."""
        shape = image.shape[-2:]
        dx = torch.randn(shape) * alpha
        dy = torch.randn(shape) * alpha
        
        dx = TF.gaussian_blur(dx.unsqueeze(0), kernel_size=11, sigma=sigma).squeeze(0)
        dy = TF.gaussian_blur(dy.unsqueeze(0), kernel_size=11, sigma=sigma).squeeze(0)
        
        x, y = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]))
        indices = torch.stack([y + dy, x + dx])
        
        return TF.grid_sample(
            image.unsqueeze(0),
            indices.permute(1, 2, 0).unsqueeze(0),
            mode='bilinear',
            padding_mode='border'
        ).squeeze(0)

    def _augment_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """Apply a single random augmentation to the image."""
        if random.random() > self.prob_augment:
            return image, "none"

        # Choose random augmentation based on weights
        aug_type = random.choices(
            list(self.augmentation_weights.keys()),
            weights=list(self.augmentation_weights.values()),
            k=1
        )[0]

        # Apply chosen augmentation
        if aug_type == AugmentationType.ROTATION:
            return self._apply_rotation(image), "rotation"
        elif aug_type == AugmentationType.SCALE:
            return self._apply_scale(image), "scale"
        elif aug_type == AugmentationType.BRIGHTNESS:
            return self._apply_brightness(image), "brightness"
        elif aug_type == AugmentationType.CONTRAST:
            return self._apply_contrast(image), "contrast"
        elif aug_type == AugmentationType.NOISE:
            return self._apply_gaussian_noise(image), "noise"
        elif aug_type == AugmentationType.ELASTIC and self.enable_elastic:
            return self._apply_elastic_transform(image), "elastic"
        
        return image, "none"

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collate function to be used with DataLoader.
        
        Args:
            batch: List of tuples (image, target)
        
        Returns:
            Tuple containing:
                - Batch of augmented images: torch.Tensor of shape [B, 1, H, W]
                - Batch of targets: torch.LongTensor of shape [B]
        """
        images, targets = zip(*batch)
        
        # Apply single random augmentation to each image
        augmented_images, aug_types = zip(*[self._augment_image(image) for image in images])
        
        # DO NOT apply any augmentation to the target but differetn class

        return torch.stack(augmented_images), torch.stack(targets)