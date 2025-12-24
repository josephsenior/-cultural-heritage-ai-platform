# 🎨 Cultural Heritage AI Platform

> **A comprehensive AI-powered platform for art authentication, generation, restoration, and cultural heritage preservation**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


https://github.com/user-attachments/assets/dfa12db7-d695-4560-a35f-70be2233d0ff


## 🌟 Overview

The **Cultural Heritage AI Platform** is an integrated system that combines state-of-the-art AI technologies to address multiple challenges in art and cultural heritage preservation. This platform provides tools for authenticating artwork, generating artistic content, restoring damaged monuments, converting 2D images to 3D models, and answering art-related questions using Retrieval-Augmented Generation (RAG).

**Note**: This repository contains 5 of the 6 platform modules. The "Fake vs Real Art Classification" module is maintained separately and not included in this repository.

### 🎯 Key Capabilities

- **🔍 Art Authentication**: Distinguish between AI-generated and human-created artwork with 91% accuracy
- **🎨 Artistic Image Generation**: Generate images in the style of famous artists using Stable Diffusion
- **🏛️ Heritage Restoration**: Restore damaged monuments and statues using fine-tuned Stable Diffusion XL
- **📐 2D to 3D Conversion**: Transform 2D images of statues into realistic 3D meshes
- **💬 Art Q&A System**: Answer art-related questions using RAG with semantic search

---

## 🏗️ Platform Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Cultural Heritage AI Platform                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Art        │  │  Artistic    │  │  Heritage    │          │
│  │Authentication│  │   Image      │  │ Restoration  │          │
│  │   Module     │  │  Generation  │  │   Module     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │   2D to 3D   │  │   Art Q&A    │                             │
│  │  Conversion  │  │   RAG System  │                             │
│  │   Module     │  │    Module    │                             │
│  └──────────────┘  └──────────────┘                             │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         Shared AI Infrastructure & Models                  │   │
│  │  • Stable Diffusion XL  • Vision Transformers             │   │
│  │  • LoRA/PEFT Fine-tuning  • Embedding Models              │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

For detailed architecture documentation, see [ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 📦 Platform Modules

**Total Platform Modules**: 6 (5 included in this repository)

### 1. 🔍 Art Authentication Module
**Location**: `Art-Authentication-Sport-Fake-From-Fake-AI-Generated-Art/`

Detects AI-generated artwork vs. human-created art using multiple deep learning architectures.

**Key Features**:
- Multiple model architectures (CNN, ViT, Swin Transformer, ResNet50, Hybrid models)
- **Best Performance**: Swin Transformer (91% test accuracy)
- Data augmentation pipeline
- Comprehensive evaluation metrics

**Use Cases**:
- Art market authentication
- Digital art verification
- Museum collection validation

### 2. 🎨 Artistic Image Generation Module
**Location**: `Artistic-Image-Generator-Inspired-By-Famous-Artists/`

Generates artistic images in the style of famous artists using Stable Diffusion with RAG-enhanced prompts.

**Key Features**:
- Text-to-image generation with artist style transfer
- FAISS-based semantic search for style retrieval
- Smart prompt fusion combining user input with artist characteristics
- LoRA fine-tuning support for custom styles

**Use Cases**:
- Creative art generation
- Educational art style exploration
- Monument visualization in artistic styles

### 3. 🏛️ Heritage Restoration Module
**Location**: `Lost-Heritage-Restoration-Task-With-MultiModal-Conditional-Diffusion-Modals-Lora-PEFT-/`

Restores damaged monuments and statues using fine-tuned Stable Diffusion XL with depth conditioning.

**Key Features**:
- Fine-tuned Stable Diffusion XL with LoRA + PEFT
- Multi-modal conditioning (depth maps, captions, damage masks)
- Automated data pipeline from Wikimedia Commons, Smithsonian, Europeana
- 40,000+ paired training samples (damaged → restored)
- High-fidelity 1024×1024 output resolution

**Use Cases**:
- Cultural heritage preservation
- Archaeological restoration
- Museum artifact reconstruction

### 4. 📐 2D to 3D Conversion Module
**Location**: `2d-to-3d-conversion-with-Hyunyan-3d-finetuned-on-Art-statues-Pipeline-/`

Converts 2D images of statues into realistic 3D meshes using Hunyuan3D diffusion model.

**Key Features**:
- Single-image 3D reconstruction
- GLB format output (AR/VR compatible)
- High-quality mesh generation
- Integration with 3D software (Blender, Unity)

**Use Cases**:
- 3D digitization of cultural artifacts
- AR/VR applications
- Virtual museum exhibitions

### 5. 💬 Art Q&A RAG System
**Location**: `AI-RAG-Agent-Answering-Art-related-questions/`

Answers art-related questions using Retrieval-Augmented Generation with semantic search.

**Key Features**:
- RAG architecture with semantic embeddings
- Image processing and analysis pipeline
- Stable Diffusion integration for visual answers
- Comprehensive art knowledge base

**Use Cases**:
- Museum visitor assistance
- Art education platforms
- Research and documentation

### 6. 🎭 Fake vs Real Art Classification Module
**Status**: Separate repository (not included in this platform)

A specialized module for classifying fake vs. real artwork. This module is maintained independently and complements the other platform modules.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for all modules)
- 16GB+ RAM (32GB recommended for training)
- 50GB+ free disk space for models and datasets

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/josephsenior/cultural-heritage-ai-platform.git
cd cultural-heritage-ai-platform
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables** (create `.env` file):
```bash
HUGGINGFACE_TOKEN=your_token_here
CUDA_VISIBLE_DEVICES=0
```

For detailed setup instructions, see [GETTING_STARTED.md](docs/GETTING_STARTED.md).

---

## 📚 Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)** - Detailed system architecture and design decisions
- **[Getting Started](docs/GETTING_STARTED.md)** - Step-by-step setup and usage guide
- **[Module Guides](docs/modules/)** - Individual module documentation
  - [Art Authentication Guide](docs/modules/art-authentication.md)
  - [Image Generation Guide](docs/modules/image-generation.md)
  - [Heritage Restoration Guide](docs/modules/heritage-restoration.md)
  - [2D to 3D Conversion Guide](docs/modules/2d-to-3d.md)
  - [RAG Q&A System Guide](docs/modules/rag-qa.md)

---

## 🎓 Technical Highlights

### Model Performance

| Module | Model | Accuracy/Metric | Notes |
|--------|-------|-----------------|-------|
| Art Authentication | Swin Transformer | **91%** test accuracy | Best performing model |
| Heritage Restoration | Stable Diffusion XL + LoRA | High fidelity | 1024×1024 output |
| Image Generation | Stable Diffusion 1.0 | High quality | Artist style transfer |
| 2D to 3D | Hunyuan3D-DiT | Realistic meshes | GLB format output |

### Technologies Used

- **Deep Learning**: PyTorch, Transformers, Diffusers
- **Computer Vision**: Vision Transformers, CNNs, Swin Transformers
- **Generative AI**: Stable Diffusion XL, LoRA, PEFT
- **3D Processing**: Hunyuan3D, Trimesh
- **NLP/RAG**: Sentence Transformers, FAISS, LangChain
- **Image Processing**: Real-ESRGAN, Waifu2x, DPT Hybrid

---

## 🔬 Research & Innovation

This platform demonstrates several innovative approaches:

1. **Multi-Modal Heritage Restoration**: Combining depth maps, captions, and damage masks for realistic restoration
2. **Efficient Fine-Tuning**: Using LoRA and PEFT for parameter-efficient model adaptation
3. **RAG-Enhanced Generation**: Semantic search for style-aware image generation
4. **Hybrid Authentication**: Combining multiple architectures for robust art verification

---

## 📊 Dataset Information

- **Art Authentication**: 130,000 images (100K train, 30K test)
- **Heritage Restoration**: 40,000+ image triplets from multiple sources
- **Image Generation**: Artist style dataset with descriptions
- **RAG System**: Art knowledge base with semantic embeddings

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution

- Model improvements and optimizations
- Additional artist styles for generation
- New restoration techniques
- Documentation improvements
- Bug fixes and testing

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: Some datasets used are under respective Creative Commons and institutional licenses (e.g., Europeana, Wikimedia Commons, Smithsonian Museum).

---

## 🙏 Acknowledgments

Special thanks to:

- Hugging Face for model hosting and libraries
- Tencent for Hunyuan3D model
- Stability AI for Stable Diffusion
- Wikimedia Commons, Smithsonian Museum, Europeana for datasets
- The open-source community for various tools and libraries

---

## 📧 Contact & Support

- **GitHub Issues**: [Open an issue](https://github.com/josephsenior/cultural-heritage-ai-platform/issues)
- **Email**: [Your email]
- **LinkedIn**: [Your LinkedIn profile]

---

## 🎥 Demo Video

*Demo video coming soon!*

---

## ⭐ Star History

If you find this project useful, please consider giving it a star ⭐!

---

**Built with ❤️ for preserving and understanding cultural heritage**

