# llama.cpp

![llama](https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp)](https://github.com/ggml-org/llama.cpp/releases)
[![Server](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)

[Manifesto](https://github.com/ggml-org/llama.cpp/discussions/205) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md)


## Converting Models from HuggingFace

If you want to convert a model directly from HuggingFace to GGUF format (e.g., embedding models like VoyageAI's models), follow these steps:

### Prerequisites

1. Make sure all required files are present (restore if using a cloned repo):
   ```bash
   git restore .
   ```

2. Install Python dependencies with `uv` (or use `pip`):
   ```bash
   # Using uv (recommended)
   uv pip install -r requirements.txt --index-strategy unsafe-best-match

   # Or using pip
   pip install -r requirements.txt
   ```

### Converting a Model

Use the `convert_hf_to_gguf.py` script with the `--remote` flag to convert directly from HuggingFace:

```bash
# Example: Converting voyageai/voyage-4-nano (embedding model)
uv run convert_hf_to_gguf.py --remote voyageai/voyage-4-nano --embedding

# Or with python directly
python convert_hf_to_gguf.py --remote voyageai/voyage-4-nano --embedding

# For regular language models (no --embedding flag needed)
uv run convert_hf_to_gguf.py --remote HuggingFaceTB/SmolLM2-1.7B-Instruct
```

The script will:
- Download the model config and tokenizer from HuggingFace
- Stream the safetensors weights remotely without downloading to disk
- Convert to GGUF format with proper pooling configuration (if `--embedding` is used)
- Output a file like `voyageai-voyage-4-nano-bf16.gguf`

**Important:** Use the `--embedding` flag for embedding models (like VoyageAI, sentence-transformers models, etc.) to properly configure the pooling type for embedding extraction.

### Using with Ollama

After converting, you can import and use the model with Ollama:

#### Import the Model

```bash
# Create a Modelfile
cat > Modelfile << EOF
FROM ./voyageai-voyage-4-nano-bf16.gguf
EOF

# Import into Ollama
ollama create voyage-4-nano -f Modelfile
```

#### Use for Embeddings

```bash
# Generate embeddings via API
curl http://localhost:11434/api/embed -d '{
  "model": "voyage-4-nano",
  "input": "Your text to embed here"
}'

# Multiple texts
curl http://localhost:11434/api/embed -d '{
  "model": "voyage-4-nano",
  "input": ["First text", "Second text", "Third text"]
}'
```

#### Python Example

```python
import ollama
import numpy as np

# Single embedding
response = ollama.embed(model='voyage-4-nano', input='Hello, world!')
embedding = response['embeddings'][0]
print(f"Embedding dimension: {len(embedding)}")  # 2048 for voyage-4-nano

# Similarity search
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

query = "What is machine learning?"
docs = [
    "Machine learning is a subset of AI",
    "I love eating pizza",
    "Neural networks are used in deep learning"
]

query_emb = ollama.embed(model='voyage-4-nano', input=query)['embeddings'][0]
doc_embs = ollama.embed(model='voyage-4-nano', input=docs)['embeddings']

for doc, emb in zip(docs, doc_embs):
    sim = cosine_similarity(query_emb, emb)
    print(f"{sim:.4f} - {doc}")
```

### Additional Options

```bash
# Convert with specific output type
uv run convert_hf_to_gguf.py --remote MODEL_ID --outtype f16

# Convert from local directory
uv run convert_hf_to_gguf.py /path/to/model

# See all options
uv run convert_hf_to_gguf.py --help
```

**Note:** For gated models, set your HuggingFace token:
```bash
export HF_TOKEN=your_token_here
```

