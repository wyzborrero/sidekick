"""
Configuration settings for Sidekick.
"""

import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class SidekickConfig:
    """Main configuration class for Sidekick."""
    
    # Model settings
    default_model_path: str = ""
    lora_models_path: str = "./models/lora"
    checkpoint_path: str = "./checkpoints"
    
    # Output settings
    output_path: str = "./output"
    temp_path: str = "./temp"
    
    # Training settings
    default_learning_rate: float = 1e-4
    default_batch_size: int = 4
    default_epochs: int = 10
    default_rank: int = 16
    
    # Generation settings
    default_width: int = 512
    default_height: int = 512
    default_steps: int = 20
    default_cfg_scale: float = 7.5
    
    # Video settings
    default_fps: int = 30
    default_duration: float = 2.0
    
    # Processing settings
    max_image_size: int = 2048
    enable_gpu: bool = True
    gpu_memory_fraction: float = 0.8
    
    # UI settings
    show_advanced_options: bool = False
    auto_save_outputs: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SidekickConfig':
        """Create config from dictionary."""
        return cls(**data)

def get_config_path() -> str:
    """Get the path to the configuration file."""
    return os.path.join(os.path.dirname(__file__), "..", "sidekick_config.json")

def load_config() -> SidekickConfig:
    """Load configuration from file."""
    config_path = get_config_path()
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            return SidekickConfig.from_dict(data)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"Error loading config: {e}. Using default configuration.")
    
    return SidekickConfig()

def save_config(config: SidekickConfig) -> bool:
    """Save configuration to file."""
    config_path = get_config_path()
    
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False
