"""
Image generation node with LoRA support.
"""

import torch
from typing import Dict, Any, Tuple
from ..base import SidekickImageNode

class SidekickImageGeneratorNode(SidekickImageNode):
    """Enhanced image generation node with LoRA support."""
    
    CATEGORY = "sidekick/generation"
    DISPLAY_NAME = "Sidekick Image Generator"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "generation_info")
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg_scale": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**32 - 1}),
            },
            "optional": {
                "lora_model": ("LORA_MODEL",),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }
    
    def execute(self, model, prompt, width, height, steps, cfg_scale, seed,
                lora_model=None, lora_strength=1.0, negative_prompt="") -> Tuple:
        """Generate image with optional LoRA."""
        
        # Placeholder implementation
        generation_info = f"Generation Parameters:\n"
        generation_info += f"- Prompt: {prompt[:50]}...\n"
        generation_info += f"- Size: {width}x{height}\n"
        generation_info += f"- Steps: {steps}\n"
        generation_info += f"- CFG Scale: {cfg_scale}\n"
        generation_info += f"- Seed: {seed}\n"
        
        if lora_model is not None:
            generation_info += f"- LoRA Strength: {lora_strength}\n"
        
        # Create placeholder image tensor
        placeholder_image = torch.zeros((1, 3, height, width))
        
        return (placeholder_image, generation_info)
