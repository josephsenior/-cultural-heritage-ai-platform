# Art Q&A RAG System Guide

## Overview

Answer art-related questions using Retrieval-Augmented Generation (RAG) with semantic search. The system combines text embeddings, FAISS similarity search, and generative models to provide accurate, context-aware answers about art.

## Architecture

```
User Question
    ↓
Query Embedding (Sentence Transformer)
    ↓
FAISS Semantic Search
    ↓
Retrieve Relevant Context
    ├─→ Text Context
    └─→ Image Context (optional)
         ↓
RAG Pipeline
    ├─→ LLM Generation (with context)
    └─→ Image Generation (if needed)
         ↓
Combined Answer
```

## Key Components

### 1. Embedding Model
- **Model**: Sentence Transformer
- **Purpose**: Convert questions and documents to embeddings
- **Dimension**: 384 (all-MiniLM-L6-v2)

### 2. Vector Store
- **Technology**: FAISS (Facebook AI Similarity Search)
- **Index Type**: IndexFlatL2
- **Purpose**: Fast semantic search

### 3. Knowledge Base
- **Content**: Art descriptions, historical data, style information
- **Format**: Text documents with embeddings
- **Size**: Varies based on dataset

### 4. Generation Models
- **Text**: LLM (via transformers)
- **Images**: Stable Diffusion (for visual answers)

## Usage

### Basic Q&A

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAISS index (pre-built)
index = faiss.read_index("art_knowledge_base.index")

# User question
question = "What is the style of Leonardo da Vinci's paintings?"

# Encode question
question_embedding = embedding_model.encode([question])

# Search for relevant context
k = 5  # Top 5 results
distances, indices = index.search(question_embedding, k)

# Retrieve context documents
contexts = [knowledge_base[i] for i in indices[0]]

# Generate answer (using LLM with context)
answer = generate_answer(question, contexts)
print(answer)
```

### With Image Generation

```python
# If question requires visual answer
if needs_visual_answer(question):
    # Generate image
    image = generate_art_image(answer_prompt)
    # Return both text and image
    return {"text": answer, "image": image}
```

## Notebooks

- `description.ipynb` - Main RAG pipeline
- `img_pross.ipynb` - Image processing
- `new_data.ipynb` - Data preparation
- `stable_diffusion_model.ipynb` - Image generation integration

## Knowledge Base Setup

### Building the Index

```python
# Load art descriptions
art_descriptions = load_art_descriptions()

# Encode descriptions
embeddings = embedding_model.encode(art_descriptions)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype('float32'))

# Save index
faiss.write_index(index, "art_knowledge_base.index")
```

### Adding New Knowledge

1. Add new documents to knowledge base
2. Encode new documents
3. Add to FAISS index
4. Save updated index

## Query Types

### Supported Questions

- **Style Questions**: "What is the style of X artist?"
- **Historical Questions**: "When was X artwork created?"
- **Technical Questions**: "What techniques did X use?"
- **Comparison Questions**: "Compare X and Y artists"
- **Visual Questions**: "Show me an example of X style"

### Query Enhancement

- **Query Expansion**: Add related terms
- **Re-ranking**: Improve result relevance
- **Context Filtering**: Filter by art type, era, etc.

## Performance

- **Query Time**: ~1 second per question
- **Memory**: ~2GB RAM
- **Accuracy**: High with good knowledge base

## Tips

1. **Good Knowledge Base**: Quality data = quality answers
2. **Query Clarity**: Clear questions get better answers
3. **Context Size**: Retrieve 3-5 relevant contexts
4. **Answer Synthesis**: Combine multiple contexts
5. **Visual Answers**: Use image generation when helpful

## Use Cases

- Museum visitor assistance
- Art education platforms
- Research and documentation
- Art history queries
- Style identification

## Future Enhancements

- Multi-modal RAG (text + images)
- Real-time knowledge updates
- Multi-language support
- Conversational interface
- Citation generation

## Integration

### With Other Modules

- **Image Generation**: Generate examples for answers
- **Authentication**: Verify artwork in answers
- **Restoration**: Show restoration examples

## Troubleshooting

### Common Issues

**1. Poor Answers**
- Improve knowledge base quality
- Increase context retrieval (k)
- Enhance query processing

**2. Slow Search**
- Use GPU-accelerated FAISS
- Optimize index size
- Cache frequent queries

**3. Missing Context**
- Expand knowledge base
- Improve document indexing
- Add more art descriptions

## Resources

- **FAISS Documentation**: https://github.com/facebookresearch/faiss
- **Sentence Transformers**: https://www.sbert.net/
- **RAG Papers**: [Link to relevant papers]

