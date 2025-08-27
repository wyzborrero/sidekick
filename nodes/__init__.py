"""
Node registry and imports for all Sidekick nodes.
"""

from .base import SidekickBaseNode
from .lora_training import *
from .image_generation import *
from .line_art_processing import *
from .comparison import *
from .video_output import *

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def register_node(cls, display_name=None):
    """Register a node class with ComfyUI."""
    NODE_CLASS_MAPPINGS[cls.__name__] = cls
    NODE_DISPLAY_NAME_MAPPINGS[cls.__name__] = display_name or cls.__name__
    return cls

# Auto-register all nodes that inherit from SidekickBaseNode
def _auto_register_nodes():
    """Automatically register all node classes."""
    import inspect
    import sys
    
    current_module = sys.modules[__name__]
    
    for name, obj in inspect.getmembers(current_module):
        if (inspect.isclass(obj) and 
            issubclass(obj, SidekickBaseNode) and 
            obj != SidekickBaseNode and
            hasattr(obj, 'CATEGORY')):
            register_node(obj, getattr(obj, 'DISPLAY_NAME', name))

_auto_register_nodes()
