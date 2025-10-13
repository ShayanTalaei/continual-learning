# Tokasaurus Setup Guide

This guide explains how to set up and use Tokasaurus for local inference with open-source language models.

## Overview

Tokasaurus is a high-performance inference engine for open-source language models. It provides:
- Fast inference with GPU acceleration
- Support for multiple GPUs (data/pipeline parallelism)
- OpenAI-compatible API for easy integration
- Support for various HuggingFace models

## Installation

Install Tokasaurus in your conda environment:

```bash
conda activate continual_learning
pip install tokasaurus
```

## Server Setup

### Basic Server Launch

Start a Tokasaurus server with a specific model:

```bash
toka model=meta-llama/Llama-3.2-3B-Instruct port=8080
```

### Multi-GPU Configuration

For faster inference with multiple GPUs:

**2 GPUs (Data Parallel):**
```bash
CUDA_VISIBLE_DEVICES=0,1 toka model=meta-llama/Llama-3.2-3B-Instruct dp_size=2 port=8080
```

**4 GPUs (2× Data Parallel × 2-stage Pipeline):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 toka model=meta-llama/Llama-3.2-3B-Instruct dp_size=2 pp_size=2 port=8080
```

### Performance Tuning

Additional parameters for optimizing throughput:

```bash
toka model=meta-llama/Llama-3.2-3B-Instruct \
     port=8080 \
     max_seqs_per_forward=128 \
     max_topk_logprobs=20 \
     kv_cache_num_tokens='(200000)' \
     dp_size=2
```

**Parameters:**
- `dp_size`: Data parallel size (number of GPUs for data parallelism)
- `pp_size`: Pipeline parallel size (number of pipeline stages)
- `max_seqs_per_forward`: Maximum sequences processed per forward pass
- `kv_cache_num_tokens`: KV cache size (affects memory usage)
- `max_topk_logprobs`: Maximum top-k log probabilities for sampling

## Configuration

### YAML Configuration

Use the `toka:` prefix in your model configuration to route to Tokasaurus:

```yaml
agent:
  type: memoryless_agent
  lm_config:
    model: "toka:meta-llama/Llama-3.2-3B-Instruct"
    temperature: 0.0
    max_output_tokens: 8192
    log_calls: true
    # Tokasaurus-specific options
    base_url: "http://localhost:8080"
    protocol: "openai"  # or "toka" for native API
    stop_sequences: ["FEEDBACK", "OBSERVATION"]
```

### Supported Models

Any HuggingFace model compatible with Tokasaurus, including:
- `meta-llama/Llama-3.2-3B-Instruct`
- `meta-llama/Llama-3.2-1B-Instruct`
- `Qwen/Qwen3-4B`
- And many others

## API Protocols

The Tokasaurus client supports two protocols:

### OpenAI-Compatible Protocol (Default)
- Endpoint: `POST /v1/chat/completions`
- Uses standard OpenAI message format
- Set `protocol: "openai"` in config

### Native Tokasaurus Protocol
- Endpoint: `POST /generate`
- Uses simplified JSON format
- Set `protocol: "toka"` in config

## Troubleshooting

### Server Not Responding
1. Check if the server is running: `curl http://localhost:8080/ping`
2. Verify GPU availability: `nvidia-smi`
3. Check port conflicts: `netstat -tulpn | grep 8080`

### Out of Memory Errors
- Reduce `kv_cache_num_tokens`
- Decrease `max_seqs_per_forward`
- Use fewer GPUs or smaller model

### Slow Inference
- Increase `dp_size` for data parallelism
- Use `pp_size` for pipeline parallelism
- Increase `max_seqs_per_forward` for better batching

## Example Usage

1. **Start the server:**
```bash
CUDA_VISIBLE_DEVICES=0,1 toka model=meta-llama/Llama-3.2-3B-Instruct dp_size=2 port=8080
```

2. **Run with configuration:**
```bash
python -m your_entrypoint --config configs/finer/memoryless_toka_llama3b.yaml
```

3. **Monitor performance:**
```bash
# Check GPU usage
nvidia-smi

# Check server health
curl http://localhost:8080/ping
```

## Integration Details

The Tokasaurus integration includes:
- `TokasaurusClient` implementing the `LanguageModel` interface
- `TokasaurusConfig` extending `LMConfig` with server-specific options
- Automatic routing via `lm_factory.get_lm_client()` when model starts with `toka:`
- Retry logic with exponential backoff
- Comprehensive logging and metrics collection

See `src/lm/tokasaurus_client.py` for implementation details.
