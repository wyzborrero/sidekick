"""
Image processing utilities.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional

def resize_image(image: torch.Tensor, target_size: Tuple[int, int], 
                mode: str = 'bilinear') -> torch.Tensor:
    """Resize image tensor to target size."""
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    
    resized = F.interpolate(image, size=target_size, mode=mode, align_corners=False)
    return resized

def normalize_image(image: torch.Tensor, mean: Optional[Tuple[float, ...]] = None,
                   std: Optional[Tuple[float, ...]] = None) -> torch.Tensor:
    """Normalize image tensor."""
    if mean is None:
        mean = (0.485, 0.456, 0.406)  # ImageNet defaults
    if std is None:
        std = (0.229, 0.224, 0.225)   # ImageNet defaults
    
    # Ensure image is in [0, 1] range
    if image.max() > 1.0:
        image = image / 255.0
    
    # Apply normalization
    mean_tensor = torch.tensor(mean).view(-1, 1, 1)
    std_tensor = torch.tensor(std).view(-1, 1, 1)
    
    if image.device != mean_tensor.device:
        mean_tensor = mean_tensor.to(image.device)
        std_tensor = std_tensor.to(image.device)
    
    normalized = (image - mean_tensor) / std_tensor
    return normalized

def denormalize_image(image: torch.Tensor, mean: Optional[Tuple[float, ...]] = None,
                     std: Optional[Tuple[float, ...]] = None) -> torch.Tensor:
    """Denormalize image tensor."""
    if mean is None:
        mean = (0.485, 0.456, 0.406)
    if std is None:
        std = (0.229, 0.224, 0.225)
    
    mean_tensor = torch.tensor(mean).view(-1, 1, 1)
    std_tensor = torch.tensor(std).view(-1, 1, 1)
    
    if image.device != mean_tensor.device:
        mean_tensor = mean_tensor.to(image.device)
        std_tensor = std_tensor.to(image.device)
    
    denormalized = image * std_tensor + mean_tensor
    return torch.clamp(denormalized, 0, 1)

def ensure_batch_dimension(image: torch.Tensor) -> torch.Tensor:
    """Ensure image has batch dimension."""
    if len(image.shape) == 3:
        return image.unsqueeze(0)
    return image

def remove_batch_dimension(image: torch.Tensor) -> torch.Tensor:
    """Remove batch dimension if batch size is 1."""
    if len(image.shape) == 4 and image.shape[0] == 1:
        return image.squeeze(0)
    return image
