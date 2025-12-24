# Art Authentication Module Guide

## Overview

The Art Authentication Module distinguishes between AI-generated and human-created artwork using advanced deep learning architectures. This module achieved **91% test accuracy** using a Swin Transformer model.

## Architecture

The module implements multiple architectures for comparison:

- **CNN**: Convolutional Neural Network baseline
- **ViT**: Vision Transformer
- **Swin Transformer**: Hierarchical Vision Transformer (Best performer)
- **ResNet50**: Residual Network
- **Hybrid CNN+ViT**: Combined architecture

## Dataset

- **Training**: 100,000 images (50K AI, 50K Human)
- **Testing**: 30,000 images (10K Human, 20K AI)
- **Format**: JPG images
- **Preprocessing**: Data augmentation (rotation, scaling, brightness)

## Usage

### Basic Usage

```python
import torch
from torchvision import transforms
from PIL import Image

# Load model (example - adjust based on your notebook)
model = load_swin_transformer_model()
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Load and classify
image = Image.open("test_art.jpg")
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(image_tensor)
    prediction = torch.softmax(output, dim=1)
    
    if prediction[0][1] > 0.5:
        print("AI-generated")
    else:
        print("Human-created")
```

### Training

See the training notebooks:
- `Vit_hybride.ipynb` - Vision Transformer training
- `swin-transformer-swin-t.ipynb` - Swin Transformer (best model)
- `resnet50v1 (2).ipynb` - ResNet50 training

## Model Performance

| Model | Train Accuracy | Test Accuracy | Precision | Recall |
|-------|---------------|---------------|-----------|--------|
| CNN | 85% | 83% | 82% | 85% |
| ViT | 88% | 87% | 86% | 88% |
| CNN+ViT | 90% | 89% | 89% | 90% |
| **Swin Transformer** | **92%** | **91%** | **90%** | **92%** |
| ResNet50 | 86% | 85% | 83% | 86% |

## Key Features

1. **Data Augmentation**: Improves generalization
2. **Multiple Architectures**: Compare different approaches
3. **Comprehensive Evaluation**: Accuracy, precision, recall metrics
4. **Production Ready**: Best model achieves 91% accuracy

## Notebooks

- `Vit_hybride.ipynb` - Hybrid ViT model
- `swin-transformer-swin-t.ipynb` - Swin Transformer (recommended)
- `resnet50v1 (2).ipynb` - ResNet50
- `efficientnetacc66.ipynb` - EfficientNet
- `hybridearchitecture.ipynb` - Hybrid architecture
- `artauth_cnn.ipynb` - CNN baseline

## Tips

1. Use Swin Transformer for best results
2. Ensure proper image preprocessing
3. Use data augmentation during training
4. Fine-tune hyperparameters for your dataset

## Future Improvements

- Ensemble of multiple models
- Real-time inference API
- Confidence score calibration
- Support for more art styles

