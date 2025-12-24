# Artistic Image Generation Module Guide

## Overview

Generate artistic images in the style of famous artists using Stable Diffusion enhanced with Retrieval-Augmented Generation (RAG). The system uses semantic search to retrieve artist-specific style descriptions and fuses them with user prompts.

## Architecture

```
User Prompt + Artist Name
    ↓
Sentence Transformer Embedding
    ↓
FAISS Similarity Search
    ↓
Retrieve Top-K Style Descriptions
    ↓
Prompt Fusion
    ↓
Stable Diffusion Generation
    ↓
Generated Image
```

## Key Components

### 1. Embedding Model
- **Model**: `all-MiniLM-L6-v2` (Sentence Transformer)
- **Purpose**: Convert text descriptions to embeddings
- **Dimension**: 384

### 2. FAISS Index
- **Type**: IndexFlatL2 (Euclidean distance)
- **Purpose**: Fast similarity search
- **Data**: Artist style descriptions

### 3. Generation Model
- **Base**: Stable Diffusion 1.0
- **Variant**: `dreamlike-diffusion-1.0` (art-optimized)
- **Resolution**: 512×512 (default)

### 4. Fine-tuning (Optional)
- **Method**: LoRA (Low-Rank Adaptation)
- **Config**: r=8, alpha=16
- **Purpose**: Custom style adaptation

## Usage

### Basic Generation

```python
from diffusers import StableDiffusionPipeline
from sentence_transformers import SentenceTransformer
import faiss
import torch

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
pipe = StableDiffusionPipeline.from_pretrained(
    "dreamlike-art/dreamlike-diffusion-1.0",
    torch_dtype=torch.float16
).to(device)

# User input
user_prompt = "A majestic tree with intricate details"
artist_name = "Leonardo da Vinci"

# Retrieve similar style descriptions
retrieved_chunks = retrieve_similar_chunks(artist_name, top_k=5)

# Create fused prompt
final_prompt = create_final_prompt(user_prompt, artist_name, retrieved_chunks)

# Generate image
image = pipe(final_prompt, num_inference_steps=50).images[0]
image.save("generated_art.png")
```

### With Custom Dataset

1. Prepare artist descriptions in JSON format
2. Build FAISS index from descriptions
3. Use retrieval for prompt enhancement

## Artist Styles

Available artists (see `artist_descriptions.json`):
- Leonardo da Vinci
- Vincent van Gogh
- Pablo Picasso
- Claude Monet
- And more...

### Adding New Artists

1. Collect style descriptions
2. Add to `artist_descriptions.json`
3. Rebuild FAISS index
4. Use in generation

## Notebooks

- `generate-new-paintings-from-different-artists-v2 (1).ipynb` - Main generation notebook
- `generate-new-paintings-from-different-artists (2).ipynb` - Alternative version
- `generateimageWithStyle.ipynb` - Style-focused generation

## Parameters

### Generation Parameters

- **num_inference_steps**: 50 (default, higher = better quality)
- **guidance_scale**: 7.5 (default, higher = more prompt adherence)
- **height/width**: 512 (default resolution)

### Retrieval Parameters

- **top_k**: 5 (number of similar descriptions to retrieve)
- **similarity_threshold**: Optional filtering

## Tips

1. **Better Prompts**: Be specific about what you want
2. **Artist Selection**: Choose artists with distinct styles
3. **Prompt Fusion**: Let the system enhance your prompt
4. **Fine-tuning**: Use LoRA for custom styles
5. **Iteration**: Try multiple generations for best results

## Performance

- **Generation Time**: ~3-5 seconds per image (GPU)
- **Memory**: ~8GB VRAM
- **Quality**: High-quality artistic outputs

## Use Cases

- Creative art generation
- Educational art style exploration
- Monument visualization in artistic styles
- Style transfer experiments

## Future Enhancements

- More artist styles
- Higher resolution outputs (1024×1024)
- Batch generation
- Style mixing
- Real-time generation API

