"""
Line art cleanup and enhancement nodes.
"""

import torch
import cv2
import numpy as np
from typing import Dict, Any, Tuple
from ..base import SidekickImageNode

class LineArtCleanupNode(SidekickImageNode):
    """Node for cleaning up line art drawings."""
    
    CATEGORY = "sidekick/line_art"
    DISPLAY_NAME = "Line Art Cleanup"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("cleaned_image", "cleanup_info")
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "noise_reduction": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "line_thickness": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),
            },
            "optional": {
                "auto_contrast": ("BOOLEAN", {"default": True}),
                "remove_artifacts": ("BOOLEAN", {"default": True}),
                "smooth_lines": ("BOOLEAN", {"default": False}),
            }
        }
    
    def execute(self, image, threshold, noise_reduction, line_thickness,
                auto_contrast=True, remove_artifacts=True, smooth_lines=False) -> Tuple:
        """Clean up line art image."""
        
        # Convert tensor to numpy for processing
        if len(image.shape) == 4:
            image = image.squeeze(0)
        
        if image.shape[0] == 3:  # CHW to HWC
            image = image.permute(1, 2, 0)
        
        img_np = (image.cpu().numpy() * 255).astype(np.uint8)
        
        # Convert to grayscale for line art processing
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        
        # Apply threshold
        _, binary = cv2.threshold(gray, int(threshold * 255), 255, cv2.THRESH_BINARY)
        
        # Noise reduction
        if noise_reduction > 0:
            kernel_size = max(1, int(noise_reduction * 5))
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Remove small artifacts
        if remove_artifacts:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = 50  # Minimum area to keep
            for contour in contours:
                if cv2.contourArea(contour) < min_area:
                    cv2.fillPoly(binary, [contour], 255)
        
        # Smooth lines
        if smooth_lines:
            binary = cv2.GaussianBlur(binary, (3, 3), 0)
            _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
        
        # Auto contrast
        if auto_contrast:
            binary = cv2.equalizeHist(binary)
        
        # Convert back to RGB tensor
        if len(binary.shape) == 2:
            binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        else:
            binary_rgb = binary
        
        # Convert back to tensor
        result_tensor = torch.from_numpy(binary_rgb.astype(np.float32) / 255.0)
        result_tensor = result_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC to BCHW
        
        cleanup_info = f"Line Art Cleanup Applied:\n"
        cleanup_info += f"- Threshold: {threshold}\n"
        cleanup_info += f"- Noise Reduction: {noise_reduction}\n"
        cleanup_info += f"- Line Thickness: {line_thickness}\n"
        cleanup_info += f"- Auto Contrast: {auto_contrast}\n"
        cleanup_info += f"- Remove Artifacts: {remove_artifacts}\n"
        cleanup_info += f"- Smooth Lines: {smooth_lines}\n"
        
        return (result_tensor, cleanup_info)
