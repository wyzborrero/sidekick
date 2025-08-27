"""
Input validation utilities.
"""

import torch
from typing import Any, Dict, List, Optional, Union

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

def validate_inputs(inputs: Dict[str, Any], requirements: Dict[str, Dict[str, Any]]) -> None:
    """Validate node inputs against requirements."""
    
    for param_name, param_config in requirements.get("required", {}).items():
        if param_name not in inputs:
            raise ValidationError(f"Required parameter '{param_name}' is missing")
        
        value = inputs[param_name]
        param_type = param_config[0] if isinstance(param_config, tuple) else param_config
        
        # Validate parameter type and constraints
        _validate_parameter(param_name, value, param_type, param_config)
    
    # Validate optional parameters if present
    for param_name, param_config in requirements.get("optional", {}).items():
        if param_name in inputs:
            value = inputs[param_name]
            param_type = param_config[0] if isinstance(param_config, tuple) else param_config
            _validate_parameter(param_name, value, param_type, param_config)

def _validate_parameter(name: str, value: Any, param_type: str, config: Any) -> None:
    """Validate a single parameter."""
    
    if param_type == "IMAGE":
        _validate_image_tensor(name, value)
    elif param_type == "MODEL":
        _validate_model(name, value)
    elif param_type == "STRING":
        _validate_string(name, value, config)
    elif param_type == "INT":
        _validate_int(name, value, config)
    elif param_type == "FLOAT":
        _validate_float(name, value, config)
    elif param_type == "BOOLEAN":
        _validate_boolean(name, value)
    elif isinstance(param_type, list):
        _validate_choice(name, value, param_type)

def _validate_image_tensor(name: str, value: Any) -> None:
    """Validate image tensor."""
    if not isinstance(value, torch.Tensor):
        raise ValidationError(f"Parameter '{name}' must be a torch.Tensor")
    
    if len(value.shape) not in [3, 4]:
        raise ValidationError(f"Parameter '{name}' must be a 3D or 4D tensor (CHW or BCHW)")
    
    if value.dtype not in [torch.float32, torch.float16]:
        raise ValidationError(f"Parameter '{name}' must be float32 or float16 tensor")

def _validate_model(name: str, value: Any) -> None:
    """Validate model parameter."""
    # Basic model validation - can be expanded
    if value is None:
        raise ValidationError(f"Parameter '{name}' cannot be None")

def _validate_string(name: str, value: Any, config: Any) -> None:
    """Validate string parameter."""
    if not isinstance(value, str):
        raise ValidationError(f"Parameter '{name}' must be a string")
    
    if isinstance(config, tuple) and len(config) > 1:
        constraints = config[1]
        if "max_length" in constraints and len(value) > constraints["max_length"]:
            raise ValidationError(f"Parameter '{name}' exceeds maximum length of {constraints['max_length']}")

def _validate_int(name: str, value: Any, config: Any) -> None:
    """Validate integer parameter."""
    if not isinstance(value, int):
        raise ValidationError(f"Parameter '{name}' must be an integer")
    
    if isinstance(config, tuple) and len(config) > 1:
        constraints = config[1]
        if "min" in constraints and value < constraints["min"]:
            raise ValidationError(f"Parameter '{name}' must be >= {constraints['min']}")
        if "max" in constraints and value > constraints["max"]:
            raise ValidationError(f"Parameter '{name}' must be <= {constraints['max']}")

def _validate_float(name: str, value: Any, config: Any) -> None:
    """Validate float parameter."""
    if not isinstance(value, (float, int)):
        raise ValidationError(f"Parameter '{name}' must be a number")
    
    if isinstance(config, tuple) and len(config) > 1:
        constraints = config[1]
        if "min" in constraints and value < constraints["min"]:
            raise ValidationError(f"Parameter '{name}' must be >= {constraints['min']}")
        if "max" in constraints and value > constraints["max"]:
            raise ValidationError(f"Parameter '{name}' must be <= {constraints['max']}")

def _validate_boolean(name: str, value: Any) -> None:
    """Validate boolean parameter."""
    if not isinstance(value, bool):
        raise ValidationError(f"Parameter '{name}' must be a boolean")

def _validate_choice(name: str, value: Any, choices: List[str]) -> None:
    """Validate choice parameter."""
    if value not in choices:
        raise ValidationError(f"Parameter '{name}' must be one of {choices}")
