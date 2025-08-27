"""
Path management utilities for Sidekick.
"""

import os
from typing import Optional
from .settings import load_config

def ensure_directory(path: str) -> str:
    """Ensure directory exists and return the path."""
    os.makedirs(path, exist_ok=True)
    return path

def get_models_path() -> str:
    """Get the models directory path."""
    config = load_config()
    return ensure_directory(config.lora_models_path)

def get_output_path() -> str:
    """Get the output directory path."""
    config = load_config()
    return ensure_directory(config.output_path)

def get_temp_path() -> str:
    """Get the temporary files directory path."""
    config = load_config()
    return ensure_directory(config.temp_path)

def get_checkpoint_path() -> str:
    """Get the checkpoint directory path."""
    config = load_config()
    return ensure_directory(config.checkpoint_path)

def get_safe_filename(filename: str) -> str:
    """Convert filename to safe format."""
    # Remove or replace unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    safe_filename = filename
    
    for char in unsafe_chars:
        safe_filename = safe_filename.replace(char, '_')
    
    # Remove multiple underscores
    while '__' in safe_filename:
        safe_filename = safe_filename.replace('__', '_')
    
    return safe_filename.strip('_')

def get_unique_filename(base_path: str, filename: str, extension: str = "") -> str:
    """Get a unique filename by adding numbers if file exists."""
    if not extension.startswith('.') and extension:
        extension = '.' + extension
    
    safe_name = get_safe_filename(filename)
    full_path = os.path.join(base_path, safe_name + extension)
    
    if not os.path.exists(full_path):
        return full_path
    
    counter = 1
    while True:
        new_name = f"{safe_name}_{counter}{extension}"
        full_path = os.path.join(base_path, new_name)
        if not os.path.exists(full_path):
            return full_path
        counter += 1
