"""
Base classes and utilities for Sidekick nodes.
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Union
from abc import ABC, abstractmethod

class SidekickBaseNode(ABC):
    """Base class for all Sidekick nodes."""
    
    CATEGORY = "sidekick"
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "execute"
    
    @classmethod
    @abstractmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Define input types for the node."""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> Tuple:
        """Execute the node's main functionality."""
        pass
    
    @classmethod
    def IS_CHANGED(cls, **kwargs) -> float:
        """Determine if node output should be recalculated."""
        return float("nan")  # Always recalculate by default

class SidekickImageNode(SidekickBaseNode):
    """Base class for image processing nodes."""
    
    @staticmethod
    def tensor_to_pil(tensor: torch.Tensor):
        """Convert tensor to PIL Image."""
        from PIL import Image
        
        # Handle different tensor formats
        if len(tensor.shape) == 4:  # Batch dimension
            tensor = tensor.squeeze(0)
        
        if tensor.shape[0] == 3:  # CHW format
            tensor = tensor.permute(1, 2, 0)
        
        # Convert to numpy and scale to 0-255
        np_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(np_image)
    
    @staticmethod
    def pil_to_tensor(image) -> torch.Tensor:
        """Convert PIL Image to tensor."""
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        tensor = transform(image)
        return tensor.unsqueeze(0)  # Add batch dimension

class SidekickVideoNode(SidekickBaseNode):
    """Base class for video processing nodes."""
    
    @staticmethod
    def frames_to_video(frames: List[torch.Tensor], fps: int = 30, output_path: str = None):
        """Convert list of frame tensors to video."""
        import cv2
        import tempfile
        import os
        
        if output_path is None:
            output_path = tempfile.mktemp(suffix='.mp4')
        
        if not frames:
            raise ValueError("No frames provided")
        
        # Get dimensions from first frame
        first_frame = frames[0]
        if len(first_frame.shape) == 4:
            first_frame = first_frame.squeeze(0)
        
        height, width = first_frame.shape[-2:]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            for frame_tensor in frames:
                # Convert tensor to numpy array
                if len(frame_tensor.shape) == 4:
                    frame_tensor = frame_tensor.squeeze(0)
                
                if frame_tensor.shape[0] == 3:  # CHW to HWC
                    frame_tensor = frame_tensor.permute(1, 2, 0)
                
                frame_np = (frame_tensor.cpu().numpy() * 255).astype(np.uint8)
                
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
        
        finally:
            out.release()
        
        return output_path
