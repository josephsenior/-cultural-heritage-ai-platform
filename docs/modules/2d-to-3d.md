# 2D to 3D Conversion Module Guide

## Overview

Convert 2D images of statues into realistic 3D meshes using Hunyuan3D, a diffusion transformer model by Tencent. Outputs are in GLB format, compatible with AR/VR applications and 3D software.

## Architecture

```
2D Statue Image
    ↓
Hunyuan3D-DiT Pipeline
    ↓
Image Encoding
    ↓
Diffusion Process (Flow Matching)
    ↓
3D Mesh Generation
    ↓
GLB Format Export
```

## Key Components

### Model
- **Name**: Hunyuan3D-2
- **Type**: Diffusion Transformer (DiT)
- **Architecture**: Flow-matching diffusion model
- **Source**: Tencent (Hugging Face)

### Output Format
- **Format**: GLB (GL Transmission Format Binary)
- **Compatibility**: Blender, Unity, Three.js, AR/VR viewers
- **Quality**: High-quality mesh with textures

## Usage

### Basic Conversion

```python
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from PIL import Image
import torch

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2'
).to(device)

# Load image
image = Image.open("statue_image.jpg")

# Generate 3D mesh
mesh = pipeline(image=image)[0]

# Export to GLB
mesh.export('statue_output.glb')
```

### Google Colab Setup

```python
# Install dependencies
!pip install hy3dgen trimesh

# Upload image
from google.colab import files
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

# Generate and download
mesh = pipeline(image=image_path)[0]
mesh.export('statue_output.glb')
files.download('statue_output.glb')
```

## Input Requirements

### Image Specifications
- **Format**: JPG, PNG
- **Recommended**: Clear, front-facing images
- **Resolution**: 512×512 or higher
- **Content**: Single statue/monument (centered)

### Best Practices
1. **Clear Images**: High contrast, well-lit
2. **Front View**: Front-facing statues work best
3. **Centered Subject**: Statue should be centered
4. **Good Resolution**: Higher resolution = better mesh
5. **Simple Background**: Plain backgrounds preferred

## Notebook

- `2dto3d.ipynb` - Main conversion notebook

## Output

### GLB File
- **Format**: Binary GLTF
- **Size**: Varies (typically 5-50MB)
- **Content**: 3D mesh with textures
- **Vertices**: High-quality mesh

### Using Output

**Blender**:
1. File → Import → glTF 2.0
2. Select your .glb file
3. Mesh appears in scene

**Unity**:
1. Drag .glb file into Assets
2. Unity automatically imports
3. Use in scenes

**Three.js**:
```javascript
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
const loader = new GLTFLoader();
loader.load('statue_output.glb', (gltf) => {
    scene.add(gltf.scene);
});
```

## Performance

- **Generation Time**: ~30 seconds per mesh (GPU)
- **Memory**: ~16GB VRAM
- **Quality**: High-quality 3D meshes

## Tips

1. **Image Quality**: Use high-resolution, clear images
2. **Lighting**: Even lighting works best
3. **Angle**: Front-facing images produce better results
4. **Patience**: Generation takes time, be patient
5. **Post-processing**: Clean up mesh in Blender if needed

## Use Cases

- 3D digitization of cultural artifacts
- AR/VR applications
- Virtual museum exhibitions
- 3D printing preparation
- Educational 3D models

## Limitations

- Works best with statues/monuments
- Single object per image
- Front-facing views preferred
- Complex scenes may not work well

## Future Enhancements

- Multi-view reconstruction
- Texture enhancement
- Mesh optimization
- Batch processing
- Real-time preview

## Troubleshooting

### Common Issues

**1. Out of Memory**
- Reduce image resolution
- Use CPU (slower but works)

**2. Poor Quality**
- Use higher resolution input
- Ensure good lighting
- Try different angles

**3. Import Errors**
- Install: `pip install hy3dgen trimesh`
- Check CUDA availability

## Resources

- **Hunyuan3D Paper**: [Link to paper]
- **Hugging Face Model**: https://huggingface.co/tencent/Hunyuan3D-2
- **GLB Format**: https://www.khronos.org/gltf/

