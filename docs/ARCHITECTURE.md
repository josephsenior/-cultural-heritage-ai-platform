# 🏗️ Platform Architecture

## Overview

The Cultural Heritage AI Platform is designed as a modular system where each component addresses a specific challenge in art and cultural heritage preservation. The platform leverages shared AI infrastructure while maintaining module independence.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Cultural Heritage AI Platform                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    User Interface Layer                         │   │
│  │  (Jupyter Notebooks / Future: Web API / Gradio Interface)      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                      │
│                                    ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Application Modules                           │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                   │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │   │
│  │  │   Art        │  │  Artistic   │  │  Heritage   │           │   │
│  │  │Authentication│  │   Image     │  │ Restoration │           │   │
│  │  │   Module     │  │ Generation  │  │   Module    │           │   │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │   │
│  │         │                 │                  │                    │   │
│  │  ┌──────┴───────┐  ┌──────┴───────┐                             │   │
│  │  │   2D to 3D   │  │   Art Q&A   │                             │   │
│  │  │  Conversion  │  │   RAG System │                             │   │
│  │  │   Module     │  │    Module   │                             │   │
│  │  └──────────────┘  └──────────────┘                             │   │
│  │                                                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                      │
│                                    ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              Shared AI Infrastructure Layer                       │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                   │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │   │
│  │  │  Vision      │  │  Generative  │  │  Embedding   │          │   │
│  │  │ Transformers │  │    Models    │  │   Models     │          │   │
│  │  │  • ViT       │  │  • SD XL     │  │  • Sentence   │          │   │
│  │  │  • Swin      │  │  • SD 1.0    │  │    Transform │          │   │
│  │  │  • ResNet    │  │  • Hunyuan3D  │  │  • FAISS      │          │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │   │
│  │                                                                   │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │   │
│  │  │ Fine-tuning  │  │  Image       │  │  Data         │          │   │
│  │  │  • LoRA      │  │  Processing  │  │  Processing  │          │   │
│  │  │  • PEFT      │  │  • Real-ESR  │  │  • Augment   │          │   │
│  │  │  • QLoRA     │  │  • Waifu2x   │  │  • Preprocess │          │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │   │
│  │                                                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                      │
│                                    ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Data & Storage Layer                          │   │
│  │  • Hugging Face Hub  • Local Storage  • Cloud Storage (Azure)   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

## Module Architecture Details

### 1. Art Authentication Module

**Architecture**: Multi-model ensemble approach

```
Input Image
    │
    ├─→ CNN Branch (ResNet50)
    ├─→ Vision Transformer Branch
    ├─→ Swin Transformer Branch
    └─→ Hybrid CNN+ViT Branch
         │
         └─→ Ensemble Voting
              │
              └─→ Output: AI/Human Classification + Confidence
```

**Key Components**:
- **Data Pipeline**: Augmentation, normalization, train/test split
- **Model Architectures**: CNN, ViT, Swin Transformer, ResNet50, Hybrid
- **Training**: Cross-entropy loss, Adam optimizer, learning rate scheduling
- **Evaluation**: Accuracy, Precision, Recall, F1-score

**Best Model**: Swin Transformer (91% test accuracy)

---

### 2. Artistic Image Generation Module

**Architecture**: RAG-enhanced Stable Diffusion

```
User Prompt + Artist Name
    │
    ├─→ Sentence Transformer Embedding
    │        │
    │        └─→ FAISS Similarity Search
    │                 │
    │                 └─→ Retrieve Top-K Style Descriptions
    │
    └─→ Prompt Fusion Engine
         │
         ├─→ User Prompt
         ├─→ Artist Name
         └─→ Retrieved Style Descriptions
              │
              └─→ Fused Prompt
                   │
                   └─→ Stable Diffusion Pipeline
                        │
                        └─→ Generated Image (Artist Style)
```

**Key Components**:
- **Embedding Model**: SentenceTransformer (all-MiniLM-L6-v2)
- **Search Engine**: FAISS IndexFlatL2
- **Generation Model**: Stable Diffusion 1.0 (dreamlike-diffusion)
- **Fine-tuning**: Optional LoRA for custom styles

---

### 3. Heritage Restoration Module

**Architecture**: Multi-modal Conditional Diffusion

```
Damaged Image
    │
    ├─→ Depth Map Generation (DPT Hybrid)
    ├─→ Image Captioning (Joy Transformer)
    ├─→ Damage Mask Detection (YOLO)
    └─→ Feature Extraction
         │
         └─→ Multi-Modal Conditioning
              │
              ├─→ Depth Map
              ├─→ Caption + Features
              ├─→ Damage Mask
              └─→ Original Image
                   │
                   └─→ Stable Diffusion XL Inpainting
                        │ (Fine-tuned with LoRA + PEFT)
                        └─→ Restored Image (1024×1024)
```

**Key Components**:
- **Base Model**: Stable Diffusion XL Inpainting
- **Fine-tuning**: LoRA (r=8, alpha=16) + PEFT
- **Conditioning**: Depth maps, captions, semantic masks
- **Preprocessing**: Real-ESRGAN (4×), Waifu2x (denoising)
- **Training Data**: 40,000+ paired samples

**Training Pipeline**:
1. Data collection from multiple sources
2. De-duplication and similarity filtering
3. Super-resolution and enhancement
4. Depth map generation
5. Caption generation
6. Damage simulation
7. Model fine-tuning with LoRA

---

### 4. 2D to 3D Conversion Module

**Architecture**: Diffusion Transformer for 3D Generation

```
2D Statue Image
    │
    └─→ Hunyuan3D-DiT Pipeline
         │
         ├─→ Image Encoding
         ├─→ Diffusion Process (Flow Matching)
         └─→ 3D Mesh Generation
              │
              └─→ GLB Format Export
```

**Key Components**:
- **Model**: Hunyuan3D-2 (Tencent)
- **Architecture**: Diffusion Transformer (DiT)
- **Output Format**: GLB (GL Transmission Format Binary)
- **Post-processing**: Mesh optimization, texture mapping

---

### 5. Art Q&A RAG System

**Architecture**: Retrieval-Augmented Generation

```
User Question
    │
    └─→ Query Embedding (Sentence Transformer)
         │
         └─→ FAISS Semantic Search
              │
              └─→ Retrieve Relevant Context
                   │
                   ├─→ Text Context
                   └─→ Image Context (if applicable)
                        │
                        └─→ RAG Pipeline
                             │
                             ├─→ LLM Generation (with context)
                             └─→ Image Generation (if needed)
                                  │
                                  └─→ Combined Answer
```

**Key Components**:
- **Embedding Model**: SentenceTransformer
- **Vector Store**: FAISS
- **Generation**: LLM (via transformers) + Stable Diffusion
- **Knowledge Base**: Art descriptions, historical data, style information

---

## Data Flow

### Training Data Flow

```
Raw Data Sources
    │
    ├─→ Wikimedia Commons
    ├─→ Smithsonian Museum API
    ├─→ Europeana API
    └─→ MIT Museum Collections
         │
         └─→ Data Pipeline
              │
              ├─→ De-duplication
              ├─→ Quality Filtering
              ├─→ Preprocessing
              │    ├─→ Super-resolution
              │    ├─→ Denoising
              │    └─→ Normalization
              │
              └─→ Feature Extraction
                   │
                   ├─→ Depth Maps
                   ├─→ Captions
                   ├─→ Embeddings
                   └─→ Metadata
                        │
                        └─→ Training Dataset
```

### Inference Data Flow

```
User Input
    │
    └─→ Module Selection
         │
         ├─→ Art Authentication
         │    └─→ Image → Model → Classification
         │
         ├─→ Image Generation
         │    └─→ Prompt → RAG → Generation → Image
         │
         ├─→ Heritage Restoration
         │    └─→ Image → Conditioning → Restoration → Image
         │
         ├─→ 2D to 3D
         │    └─→ Image → 3D Pipeline → Mesh
         │
         └─→ Q&A System
              └─→ Question → RAG → Answer
```

## Model Sharing & Reusability

The platform is designed with shared components:

1. **Embedding Models**: Used across RAG, Image Generation, and Q&A
2. **Vision Transformers**: Shared between Authentication and Restoration
3. **Stable Diffusion**: Base models shared, fine-tuned per module
4. **Preprocessing**: Common image processing utilities

## Scalability Considerations

- **Model Loading**: Lazy loading of large models
- **Caching**: FAISS indices and embeddings cached
- **Batch Processing**: Support for batch inference
- **GPU Utilization**: Efficient GPU memory management
- **Distributed Training**: Support for multi-GPU training

## Future Architecture Enhancements

1. **API Layer**: RESTful API for all modules
2. **Web Interface**: Gradio/Streamlit dashboard
3. **Model Serving**: TorchServe or TensorFlow Serving
4. **Database**: Vector database (Pinecone, Weaviate) for RAG
5. **Microservices**: Containerized modules (Docker)
6. **Orchestration**: Kubernetes for scaling

## Performance Metrics

| Module | Latency | Throughput | GPU Memory |
|--------|---------|------------|------------|
| Art Authentication | ~50ms | 20 img/s | 4GB |
| Image Generation | ~3s | 0.3 img/s | 8GB |
| Heritage Restoration | ~5s | 0.2 img/s | 12GB |
| 2D to 3D | ~30s | 0.03 mesh/s | 16GB |
| RAG Q&A | ~1s | 1 query/s | 2GB |

*Metrics measured on NVIDIA RTX 3090*

---

## Security & Privacy

- **Data Privacy**: No user data stored permanently
- **Model Security**: Signed model checkpoints
- **API Security**: Rate limiting, authentication (future)
- **Content Filtering**: NSFW filtering for generated content

---

## Deployment Architecture (Future)

```
┌─────────────┐
│   Load      │
│  Balancer   │
└──────┬──────┘
       │
   ┌───┴───┐
   │  API  │
   │Gateway│
   └───┬───┘
       │
   ┌───┴──────────────────────────┐
   │                               │
┌──┴──┐  ┌──────┐  ┌──────┐  ┌────┴──┐
│Auth │  │Image │  │Restore│  │2D-3D  │
│Module│  │Gen   │  │Module│  │Module │
└──────┘  └──────┘  └──────┘  └───────┘
```

---

For implementation details of each module, see the respective module guides in `docs/modules/`.

