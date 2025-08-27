"""
Video output and animation nodes.
"""

from .animator import ImageAnimatorNode
from .video_export import VideoExportNode
from .frame_interpolation import FrameInterpolationNode

__all__ = ["ImageAnimatorNode", "VideoExportNode", "FrameInterpolationNode"]
