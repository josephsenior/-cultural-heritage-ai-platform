# Heritage Restoration Module Guide

## Overview

Restore damaged monuments and statues using fine-tuned Stable Diffusion XL with multi-modal conditioning (depth maps, captions, damage masks). This module achieves high-fidelity restoration at 1024×1024 resolution.

## Architecture

```
Damaged Image
    ↓
┌─────────────────────────────────────┐
│  Multi-Modal Conditioning Pipeline  │
├─────────────────────────────────────┤
│  • Depth Map (DPT Hybrid)          │
│  • Caption (Joy Transformer)       │
│  • Damage Mask (YOLO)               │
│  • Structural Features              │
└─────────────────────────────────────┘
    ↓
Stable Diffusion XL Inpainting
(Fine-tuned with LoRA + PEFT)
    ↓
Restored Image (1024×1024)
```

## Key Components

### 1. Base Model
- **Model**: Stable Diffusion XL Inpainting
- **Resolution**: 1024×1024
- **Fine-tuning**: LoRA (r=8, alpha=16) + PEFT

### 2. Preprocessing Pipeline
- **Super-resolution**: Real-ESRGAN (4×)
- **Denoising**: Waifu2x
- **Depth Generation**: DPT Hybrid (MiDaS)
- **De-duplication**: Similarity filtering

### 3. Captioning
- **Model**: Joy Transformer Caption Alpha-2
- **Purpose**: Generate detailed image descriptions
- **Features**: Heritage type, style, era, materials

### 4. Damage Simulation
- **Types**: Cracks, weathering, missing parts, dirt, vandalism
- **Purpose**: Create training pairs
- **Method**: Realistic damage generation

## Dataset

- **Size**: 40,000+ image triplets
- **Sources**: 
  - Wikimedia Commons
  - Smithsonian Museum API
  - Europeana API
  - MIT Museum Collections
- **Format**: (damaged_image, depth_map, caption, restored_image)

## Usage

### Basic Restoration

```python
from diffusers import StableDiffusionXLInpaintPipeline
import torch
from PIL import Image

# Load fine-tuned model
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "your-finetuned-model",
    torch_dtype=torch.float16
).to("cuda")

# Load damaged image and mask
damaged_image = Image.open("damaged_monument.jpg")
damage_mask = Image.open("damage_mask.png")

# Generate depth map (using DPT Hybrid)
depth_map = generate_depth_map(damaged_image)

# Generate caption
caption = generate_caption(damaged_image)

# Create prompt from caption
prompt = f"Restore this monument: {caption}"

# Restore
restored = pipe(
    prompt=prompt,
    image=damaged_image,
    mask_image=damage_mask,
    depth_map=depth_map,
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]

restored.save("restored_monument.png")
```

### Training Pipeline

1. **Data Collection**: Run data loading notebooks
2. **Preprocessing**: Super-resolution, denoising
3. **Feature Extraction**: Depth maps, captions
4. **Damage Simulation**: Create training pairs
5. **Fine-tuning**: Train with LoRA + PEFT

## Notebooks

### Data Preparation
- `full image prepation pipeline (removing duplicates + similarities)+ image enhacement+image.ipynb` - Image preprocessing
- `mass depth map generation with DPT hybrid.ipynb` - Depth map generation
- `Final Image captioning.ipynb` - Caption generation
- `varied damage generation (approach 2) -data prep for realistic data preparation.ipynb` - Damage simulation

### Training
- `data loading + training pipeline(LoRa+PEFT).ipynb` - Main training pipeline

### Inference
- Use the trained model with the inference pipeline

## Training Configuration

### LoRA Configuration
```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,                    # Rank
    lora_alpha=16,          # Scaling factor
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    lora_dropout=0.1
)
```

### Training Parameters
- **Batch Size**: 1-2 (depending on GPU)
- **Learning Rate**: 1e-4
- **Epochs**: 10-20
- **Optimizer**: AdamW
- **Mixed Precision**: fp16

## Results

- **Fidelity**: High-quality restoration
- **Accuracy**: Realistic texture and structure recreation
- **Robustness**: Works on various damage types
- **Resolution**: 1024×1024 output

## Tips

1. **Quality Inputs**: Use high-quality damaged images
2. **Accurate Masks**: Precise damage masks improve results
3. **Good Captions**: Detailed captions help the model
4. **Depth Maps**: Accurate depth maps enhance 3D structure
5. **Iteration**: Multiple passes for complex damage

## Use Cases

- Cultural heritage preservation
- Archaeological restoration
- Museum artifact reconstruction
- Historical monument restoration
- Educational visualization

## Performance

- **Restoration Time**: ~5-10 seconds per image (GPU)
- **Memory**: ~12GB VRAM
- **Quality**: High-fidelity outputs

## Future Enhancements

- Higher resolution (2048×2048)
- Video restoration
- Real-time restoration
- Interactive mask editing
- Batch processing API

