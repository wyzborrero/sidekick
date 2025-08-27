# Sidekick - Custom ComfyUI Node Collection

A comprehensive collection of custom ComfyUI nodes for LoRA training, image generation, line art processing, and animation.

## Features

### 🎯 LoRA Training
- **LoRA Trainer Node**: Train custom LoRA models with configurable parameters
- **Dataset Preparation**: Automated dataset preprocessing and validation
- **Training Configuration**: Fine-tune training parameters for optimal results

### 🎨 Image Generation
- **Enhanced Image Generator**: Generate images with LoRA model support
- **Style Transfer**: Apply artistic styles to generated images
- **Advanced Controls**: Comprehensive parameter control for generation

### ✏️ Line Art Processing
- **Line Art Cleanup**: Automatically clean and enhance line drawings
- **Colorization**: Add colors to line art with AI assistance
- **Enhancement**: Improve line quality and remove artifacts

### 📊 A/B Comparison
- **Visual Comparison**: Side-by-side, overlay, and difference comparisons
- **Metrics Analysis**: Similarity and quality scoring
- **Multiple View Modes**: Grid, overlay, and difference visualizations

### 🎬 Video & Animation
- **Image Animator**: Create animations from static images
- **Video Export**: Export animations as video files
- **Frame Interpolation**: Smooth animation transitions

## Installation

1. Clone this repository into your ComfyUI custom nodes directory:
\`\`\`bash
cd ComfyUI/custom_nodes/
git clone https://github.com/your-username/sidekick-comfyui.git
\`\`\`

2. Install required dependencies:
\`\`\`bash
cd sidekick-comfyui
pip install -r requirements.txt
\`\`\`

3. Restart ComfyUI

## Node Categories

All Sidekick nodes are organized under the `sidekick` category:

- `sidekick/lora` - LoRA training and management
- `sidekick/generation` - Image generation and enhancement
- `sidekick/line_art` - Line art processing and colorization
- `sidekick/comparison` - Image comparison and analysis
- `sidekick/video` - Animation and video output

## Configuration

Sidekick uses a configuration system to manage default settings:

- Configuration file: `sidekick_config.json`
- Modify paths, default parameters, and processing options
- Auto-creates directories for models, outputs, and temporary files

## Modular Architecture

The codebase is designed for easy extension:

\`\`\`
sidekick/
├── nodes/           # Node implementations
│   ├── base.py      # Base classes
│   ├── lora_training/
│   ├── image_generation/
│   ├── line_art_processing/
│   ├── comparison/
│   └── video_output/
├── config/          # Configuration management
├── utils/           # Utility functions
└── README.md
\`\`\`

## Development

### Adding New Nodes

1. Create your node class inheriting from `SidekickBaseNode`
2. Implement required methods: `INPUT_TYPES()` and `execute()`
3. Add to appropriate category module
4. The node will be auto-registered on import

### Example Node Structure

\`\`\`python
from ..base import SidekickBaseNode

class MyCustomNode(SidekickBaseNode):
    CATEGORY = "sidekick/custom"
    DISPLAY_NAME = "My Custom Node"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
            }
        }
    
    def execute(self, input_image, strength):
        # Your processing logic here
        return (processed_image,)
\`\`\`

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- NumPy
- PIL/Pillow
- ComfyUI

## License

MIT License - see LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Roadmap

- [ ] Advanced LoRA training techniques
- [ ] Real-time preview capabilities
- [ ] Integration with popular AI models
- [ ] Enhanced video processing features
- [ ] Web UI for configuration management
- [ ] Batch processing capabilities

## Support

For issues, feature requests, or questions:
- Open an issue on GitHub
- Check the documentation
- Join the ComfyUI community discussions
