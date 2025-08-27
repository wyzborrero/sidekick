"""
Configuration management for Sidekick.
"""

from .settings import SidekickConfig, load_config, save_config
from .paths import get_models_path, get_output_path, get_temp_path

__all__ = ["SidekickConfig", "load_config", "save_config", 
           "get_models_path", "get_output_path", "get_temp_path"]
