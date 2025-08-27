"""
Image animation and video generation nodes.
"""

import torch
import math
from typing import Dict, Any, Tuple, List
from ..base import SidekickVideoNode

class ImageAnimatorNode(SidekickVideoNode):
    """Node for animating images with various effects."""
    
    CATEGORY = "sidekick/video"
    DISPLAY_NAME = "Image Animator"
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video_frames", "animation_info")
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "animation_type": (["zoom_in", "zoom_out", "pan_left", "pan_right", 
                                 "pan_up", "pan_down", "rotate", "fade", "pulse"], 
                                {"default": "zoom_in"}),
                "duration": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "fps": ("INT", {"default": 30, "min": 1, "max": 60}),
            },
            "optional": {
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),
                "easing": (["linear", "ease_in", "ease_out", "ease_in_out"], {"default": "ease_out"}),
                "loop": ("BOOLEAN", {"default": False}),
            }
        }
    
    def execute(self, image, animation_type, duration, fps, 
                intensity=1.0, easing="ease_out", loop=False) -> Tuple:
        """Generate animated frames from static image."""
        
        total_frames = int(duration * fps)
        frames = []
        
        for frame_idx in range(total_frames):
            progress = frame_idx / (total_frames - 1) if total_frames > 1 else 0
            
            # Apply easing function
            eased_progress = self._apply_easing(progress, easing)
            
            # Generate frame based on animation type
            frame = self._generate_frame(image, animation_type, eased_progress, intensity)
            frames.append(frame)
        
        # Create loop frames if requested
        if loop and len(frames) > 1:
            # Add reverse frames (excluding first and last to avoid duplicates)
            reverse_frames = frames[-2:0:-1]
            frames.extend(reverse_frames)
        
        animation_info = f"Animation Generated:\n"
        animation_info += f"- Type: {animation_type}\n"
        animation_info += f"- Duration: {duration}s\n"
        animation_info += f"- FPS: {fps}\n"
        animation_info += f"- Total Frames: {len(frames)}\n"
        animation_info += f"- Intensity: {intensity}\n"
        animation_info += f"- Easing: {easing}\n"
        animation_info += f"- Loop: {loop}\n"
        
        return (frames, animation_info)
    
    def _apply_easing(self, t: float, easing_type: str) -> float:
        """Apply easing function to progress value."""
        if easing_type == "linear":
            return t
        elif easing_type == "ease_in":
            return t * t
        elif easing_type == "ease_out":
            return 1 - (1 - t) * (1 - t)
        elif easing_type == "ease_in_out":
            return 2 * t * t if t < 0.5 else 1 - 2 * (1 - t) * (1 - t)
        else:
            return t
    
    def _generate_frame(self, image: torch.Tensor, animation_type: str, 
                       progress: float, intensity: float) -> torch.Tensor:
        """Generate a single animation frame."""
        
        if animation_type == "zoom_in":
            return self._apply_zoom(image, 1.0 + progress * intensity)
        elif animation_type == "zoom_out":
            return self._apply_zoom(image, 1.0 + intensity - progress * intensity)
        elif animation_type == "pan_left":
            return self._apply_pan(image, -progress * intensity * 0.2, 0)
        elif animation_type == "pan_right":
            return self._apply_pan(image, progress * intensity * 0.2, 0)
        elif animation_type == "pan_up":
            return self._apply_pan(image, 0, -progress * intensity * 0.2)
        elif animation_type == "pan_down":
            return self._apply_pan(image, 0, progress * intensity * 0.2)
        elif animation_type == "rotate":
            return self._apply_rotation(image, progress * intensity * 360)
        elif animation_type == "fade":
            return self._apply_fade(image, 1.0 - progress * intensity)
        elif animation_type == "pulse":
            pulse_value = 0.5 + 0.5 * math.sin(progress * math.pi * 4 * intensity)
            return self._apply_fade(image, pulse_value)
        else:
            return image
    
    def _apply_zoom(self, image: torch.Tensor, scale: float) -> torch.Tensor:
        """Apply zoom transformation."""
        if scale == 1.0:
            return image
        
        # Use interpolation for zoom
        _, _, h, w = image.shape
        new_h, new_w = int(h / scale), int(w / scale)
        
        # Crop center region
        start_h = (h - new_h) // 2
        start_w = (w - new_w) // 2
        cropped = image[:, :, start_h:start_h+new_h, start_w:start_w+new_w]
        
        # Resize back to original size
        zoomed = torch.nn.functional.interpolate(
            cropped, size=(h, w), mode='bilinear', align_corners=False
        )
        
        return zoomed
    
    def _apply_pan(self, image: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
        """Apply pan transformation."""
        if dx == 0 and dy == 0:
            return image
        
        _, _, h, w = image.shape
        
        # Calculate pixel offsets
        offset_x = int(dx * w)
        offset_y = int(dy * h)
        
        # Create shifted image
        shifted = torch.zeros_like(image)
        
        # Calculate source and destination regions
        src_x1 = max(0, -offset_x)
        src_x2 = min(w, w - offset_x)
        src_y1 = max(0, -offset_y)
        src_y2 = min(h, h - offset_y)
        
        dst_x1 = max(0, offset_x)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y1 = max(0, offset_y)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        
        if src_x2 > src_x1 and src_y2 > src_y1:
            shifted[:, :, dst_y1:dst_y2, dst_x1:dst_x2] = image[:, :, src_y1:src_y2, src_x1:src_x2]
        
        return shifted
    
    def _apply_rotation(self, image: torch.Tensor, angle: float) -> torch.Tensor:
        """Apply rotation transformation."""
        # Placeholder for rotation - would need more complex implementation
        # For now, return original image
        return image
    
    def _apply_fade(self, image: torch.Tensor, alpha: float) -> torch.Tensor:
        """Apply fade effect."""
        return image * alpha
