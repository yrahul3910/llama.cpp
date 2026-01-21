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
# Example: Converting voyageai/voyage-4-nano
uv run convert_hf_to_gguf.py --remote voyageai/voyage-4-nano

# Or with python directly
python convert_hf_to_gguf.py --remote voyageai/voyage-4-nano
```

The script will:
- Download the model config and tokenizer from HuggingFace
- Stream the safetensors weights remotely without downloading to disk
- Convert to GGUF format
- Output a file like `voyageai-voyage-4-nano-bf16.gguf`

### Importing the model into `ollama`

Once you have this file, change the filename accordingly in `Modelfile`, and run `ollama create voyage-4-nano` (or whatever name you want).

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

