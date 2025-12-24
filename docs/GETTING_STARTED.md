# 🚀 Getting Started Guide

This guide will help you set up and run the Cultural Heritage AI Platform on your local machine.

## Prerequisites

### Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended)
  - Minimum: 8GB VRAM (for inference)
  - Recommended: 16GB+ VRAM (for training)
  - Tested on: NVIDIA RTX 3090, A100
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB+ free space for models and datasets
- **CPU**: Multi-core processor (4+ cores recommended)

### Software Requirements

- **Operating System**: Linux, macOS, or Windows 10/11
- **Python**: 3.8, 3.9, 3.10, or 3.11
- **CUDA**: 11.8 or 12.1 (if using GPU)
- **Git**: For cloning the repository

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/josephsenior/cultural-heritage-ai-platform.git
cd cultural-heritage-ai-platform
```

### 2. Create Virtual Environment

**Linux/macOS**:
```bash
python -m venv venv
source venv/bin/activate
```

**Windows**:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install PyTorch

Install PyTorch with CUDA support (if you have a GPU):

**CUDA 11.8**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CPU only**:
```bash
pip install torch torchvision torchaudio
```

### 4. Install Platform Dependencies

```bash
pip install -r requirements.txt
```

### 5. Install Additional Dependencies (if needed)

For GPU-accelerated FAISS:
```bash
pip uninstall faiss-cpu
pip install faiss-gpu
```

### 6. Set Up Hugging Face

1. Create an account at [huggingface.co](https://huggingface.co)
2. Generate an access token: Settings → Access Tokens → New Token
3. Login:
```bash
huggingface-cli login
```

### 7. Environment Variables

Create a `.env` file in the root directory:

```bash
# Hugging Face Token
HUGGINGFACE_TOKEN=your_token_here

# CUDA Device
CUDA_VISIBLE_DEVICES=0

# Optional: API Keys
WIKIMEDIA_API_KEY=your_key
SMITHSONIAN_API_KEY=your_key
```

## Quick Start Examples

### Example 1: Art Authentication

```python
# Navigate to the module
cd Art-Authentication-Sport-Fake-From-Fake-AI-Generated-Art

# Open Jupyter Notebook
jupyter notebook Vit_hybride.ipynb

# Or run directly
python -m jupyter notebook Vit_hybride.ipynb
```

### Example 2: Artistic Image Generation

```python
cd Artistic-Image-Generator-Inspired-By-Famous-Artists
jupyter notebook generate-new-paintings-from-different-artists-v2\ \(1\).ipynb
```

### Example 3: Heritage Restoration

```python
cd Lost-Heritage-Restoration-Task-With-MultiModal-Conditional-Diffusion-Modals-Lora-PEFT-
jupyter notebook data\ loading\ +\ training\ pipeline\(LoRa+PEFT\).ipynb
```

## Running Modules

### Option 1: Jupyter Notebooks (Recommended for Exploration)

Each module contains Jupyter notebooks that can be run interactively:

```bash
# Start Jupyter
jupyter notebook

# Navigate to the module folder and open the desired notebook
```

### Option 2: Google Colab

For users without powerful GPUs, you can use Google Colab:

1. Upload the notebook to Google Colab
2. Mount Google Drive (optional, for saving outputs)
3. Install dependencies in the first cell
4. Run the notebook

### Option 3: Python Scripts (Future)

We're working on converting notebooks to Python scripts for easier integration.

## Module-Specific Setup

### Art Authentication Module

**Dataset Setup**:
- Download the dataset (100K training, 30K test images)
- Organize into `train_data/` and `test_data/` folders
- Update paths in the notebook

**Quick Test**:
```python
from torchvision import transforms
from PIL import Image

# Load and test an image
img = Image.open("test_image.jpg")
# Run through the model (see notebook for details)
```

### Artistic Image Generation Module

**Model Download**:
- Models are automatically downloaded from Hugging Face on first run
- Ensure you have internet connection and Hugging Face token set

**Artist Styles**:
- Check `artist_descriptions.json` for available artists
- Add custom artists by extending the JSON file

### Heritage Restoration Module

**Data Pipeline**:
- The module includes scripts for downloading from APIs
- Run the data loading notebook first to set up datasets
- Expected structure: `damaged/`, `restored/`, `depth_maps/`, `captions/`

**Model Checkpoints**:
- Pre-trained checkpoints can be downloaded from Hugging Face
- Or train from scratch using the training pipeline

### 2D to 3D Conversion Module

**Model Setup**:
```python
# The model is automatically downloaded on first use
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
```

**Input Requirements**:
- Single statue image (JPG/PNG)
- Recommended: Clear, front-facing images
- Resolution: 512×512 or higher

### RAG Q&A System

**Knowledge Base Setup**:
- The system uses embedded art descriptions
- Run the data preparation notebook first
- FAISS index is created automatically

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```
Solution: Reduce batch size or image resolution
- Set batch_size=1 in training scripts
- Use smaller image sizes for inference
```

**2. Model Download Fails**
```
Solution: Check internet connection and Hugging Face token
- Verify token: huggingface-cli whoami
- Try manual download from Hugging Face website
```

**3. Import Errors**
```
Solution: Ensure all dependencies are installed
- pip install -r requirements.txt --upgrade
- Check Python version: python --version (should be 3.8+)
```

**4. FAISS Installation Issues**
```
Solution: Install correct version for your system
- CPU: pip install faiss-cpu
- GPU: pip install faiss-gpu (requires CUDA)
```

**5. Slow Performance**
```
Solution: Enable GPU acceleration
- Check CUDA: python -c "import torch; print(torch.cuda.is_available())"
- Set device to 'cuda' in notebooks
```

### Getting Help

- **GitHub Issues**: [Open an issue](https://github.com/josephsenior/cultural-heritage-ai-platform/issues)
- **Documentation**: Check module-specific guides in `docs/modules/`
- **Community**: Join discussions in GitHub Discussions

## Next Steps

1. **Explore Modules**: Start with the module that interests you most
2. **Read Documentation**: Check `docs/modules/` for detailed guides
3. **Run Examples**: Try the example notebooks in each module
4. **Experiment**: Modify parameters and see the results
5. **Contribute**: See [CONTRIBUTING.md](../CONTRIBUTING.md)

## Performance Tips

1. **Use GPU**: Always use GPU for training and inference when available
2. **Batch Processing**: Process multiple images in batches
3. **Model Caching**: Models are cached after first download
4. **Memory Management**: Clear cache between runs: `torch.cuda.empty_cache()`
5. **Mixed Precision**: Use `torch.float16` for faster inference

## Resources

- **PyTorch Documentation**: https://pytorch.org/docs/
- **Hugging Face Docs**: https://huggingface.co/docs
- **Stable Diffusion Guide**: https://huggingface.co/docs/diffusers
- **FAISS Documentation**: https://github.com/facebookresearch/faiss

---

**Ready to start?** Pick a module and dive in! 🎨

