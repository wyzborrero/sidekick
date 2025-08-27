"""
A/B comparison node for image analysis.
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple
from ..base import SidekickImageNode

class ABComparisonNode(SidekickImageNode):
    """Node for A/B comparison of images with metrics and visualization."""
    
    CATEGORY = "sidekick/comparison"
    DISPLAY_NAME = "A/B Image Comparison"
    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT", "FLOAT")
    RETURN_NAMES = ("comparison_image", "analysis_report", "similarity_score", "quality_score")
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "comparison_type": (["side_by_side", "overlay", "difference", "grid"], {"default": "side_by_side"}),
            },
            "optional": {
                "label_a": ("STRING", {"default": "Image A"}),
                "label_b": ("STRING", {"default": "Image B"}),
                "show_metrics": ("BOOLEAN", {"default": True}),
                "show_labels": ("BOOLEAN", {"default": True}),
                "overlay_opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }
    
    def execute(self, image_a, image_b, comparison_type, label_a="Image A", 
                label_b="Image B", show_metrics=True, show_labels=True, 
                overlay_opacity=0.5) -> Tuple:
        """Create A/B comparison visualization."""
        
        # Ensure images are the same size
        if image_a.shape != image_b.shape:
            # Resize image_b to match image_a
            target_height, target_width = image_a.shape[-2:]
            image_b = torch.nn.functional.interpolate(
                image_b, size=(target_height, target_width), 
                mode='bilinear', align_corners=False
            )
        
        # Calculate similarity metrics
        similarity_score = self._calculate_similarity(image_a, image_b)
        quality_score = self._calculate_quality_score(image_a, image_b)
        
        # Create comparison visualization
        if comparison_type == "side_by_side":
            comparison_image = self._create_side_by_side(image_a, image_b)
        elif comparison_type == "overlay":
            comparison_image = self._create_overlay(image_a, image_b, overlay_opacity)
        elif comparison_type == "difference":
            comparison_image = self._create_difference(image_a, image_b)
        elif comparison_type == "grid":
            comparison_image = self._create_grid(image_a, image_b)
        else:
            comparison_image = self._create_side_by_side(image_a, image_b)
        
        # Generate analysis report
        analysis_report = f"A/B Comparison Analysis:\n"
        analysis_report += f"- Comparison Type: {comparison_type}\n"
        analysis_report += f"- Image A: {label_a}\n"
        analysis_report += f"- Image B: {label_b}\n"
        analysis_report += f"- Similarity Score: {similarity_score:.3f}\n"
        analysis_report += f"- Quality Score: {quality_score:.3f}\n"
        analysis_report += f"- Image Dimensions: {image_a.shape[-2:]}px\n"
        
        return (comparison_image, analysis_report, similarity_score, quality_score)
    
    def _calculate_similarity(self, img_a: torch.Tensor, img_b: torch.Tensor) -> float:
        """Calculate structural similarity between images."""
        # Simple MSE-based similarity (can be enhanced with SSIM later)
        mse = torch.mean((img_a - img_b) ** 2)
        similarity = 1.0 / (1.0 + mse.item())
        return similarity
    
    def _calculate_quality_score(self, img_a: torch.Tensor, img_b: torch.Tensor) -> float:
        """Calculate relative quality score."""
        # Simple variance-based quality metric
        var_a = torch.var(img_a)
        var_b = torch.var(img_b)
        quality_ratio = min(var_a, var_b) / max(var_a, var_b)
        return quality_ratio.item()
    
    def _create_side_by_side(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        """Create side-by-side comparison."""
        return torch.cat([img_a, img_b], dim=-1)  # Concatenate along width
    
    def _create_overlay(self, img_a: torch.Tensor, img_b: torch.Tensor, opacity: float) -> torch.Tensor:
        """Create overlay comparison."""
        return img_a * (1 - opacity) + img_b * opacity
    
    def _create_difference(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        """Create difference map."""
        diff = torch.abs(img_a - img_b)
        # Enhance visibility of differences
        diff = torch.clamp(diff * 3.0, 0.0, 1.0)
        return diff
    
    def _create_grid(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        """Create 2x2 grid with original and difference images."""
        diff = self._create_difference(img_a, img_b)
        overlay = self._create_overlay(img_a, img_b, 0.5)
        
        # Create 2x2 grid
        top_row = torch.cat([img_a, img_b], dim=-1)
        bottom_row = torch.cat([diff, overlay], dim=-1)
        grid = torch.cat([top_row, bottom_row], dim=-2)
        
        return grid
