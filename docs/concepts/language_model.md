## Language Model

The language model abstraction provides a unified interface for calling various LLM providers.

### Base Interface

**`LanguageModel`**: Abstract base class
- `call(system_prompt: str, user_prompt: str) -> str`: Synchronous LLM call
- `config: LMConfig`: Runtime parameters
- Automatic retry logic with exponential backoff
- Structured logging when enabled

### Supported Clients

#### GeminiClient
**Provider**: Google Gemini (via official API)

**Configuration** (`GeminiConfig`):
```yaml
lm_config:
  model: "gemini-2.5-flash"  # or "gemini-1.5-pro", etc.
  temperature: 0.7
  max_output_tokens: 2048
  log_calls: true
  thinking_budget: 8192  # Optional: enables Gemini thinking mode
  max_retries: 5
  starting_delay: 1.0
  backoff_factor: 2.0
  max_delay: 10.0
```

**Features**:
- Thinking mode support (extended reasoning)
- Automatic model routing based on `model` field containing "gemini"
- Token usage metrics collection
- Safety settings configuration

**Authentication**: Requires `GOOGLE_API_KEY` environment variable or API key in config

#### TokasaurusClient
**Provider**: Local Tokasaurus inference server (open-source models)

**Configuration** (`TokasaurusConfig`):
```yaml
lm_config:
  model: "toka:meta-llama/Llama-3.1-8B-Instruct"  # Prefix with "toka:" for routing
  temperature: 0.7
  max_output_tokens: 2048
  log_calls: true
  
  # Tokasaurus-specific options
  base_url: "http://localhost:8080"
  protocol: "openai"  # or "toka" for native API
  stop_sequences: ["FEEDBACK", "OBSERVATION"]
  timeout_s: 900.0
  enable_health_check: false
  
  # Standard retry options
  max_retries: 5
  starting_delay: 1.0
  backoff_factor: 2.0
  max_delay: 10.0
```

**Protocols**:
- `"openai"`: OpenAI-compatible API at `/v1/chat/completions` (default)
- `"toka"`: Native Tokasaurus API at `/generate`

**Features**:
- Support for any HuggingFace model compatible with Tokasaurus
- Stop sequences for early termination
- Truncation detection (warns when hitting max_output_tokens)
- Optional health check before calls (disabled by default for better performance)
- Comprehensive error logging with retry details

**Common models**:
- `meta-llama/Llama-3.1-8B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- And more (see [Tokasaurus Setup](../guides/tokasaurus-setup.md))

**Server setup**: See [Tokasaurus Setup Guide](../guides/tokasaurus-setup.md)

#### OpenAI Client
**Provider**: OpenAI API (gpt-4, gpt-3.5-turbo, etc.)

**Configuration**:
```yaml
lm_config:
  model: "gpt-4o"  # or "gpt-4o-mini", "o1-preview", etc.
  temperature: 0.7
  max_output_tokens: 2048
  log_calls: true
```

**Authentication**: Requires `OPENAI_API_KEY` environment variable

### Client Routing

The factory function `get_lm_client(lm_config)` automatically routes to the correct client:

1. **Model prefix `"toka:"`** → `TokasaurusClient`
   - Example: `"toka:meta-llama/Llama-3.1-8B-Instruct"`

2. **Model contains `"gemini"`** → `GeminiClient`
   - Example: `"gemini-2.5-flash"`, `"gemini-1.5-pro"`

3. **Model starts with `"gpt-"` or `"o1-"`** → `OpenAIClient`
   - Example: `"gpt-4o"`, `"o1-preview"`

4. **Default**: Attempts OpenAI client

### Common Configuration Fields

All LM configs inherit from `LMConfig`:

```yaml
lm_config:
  model: str                    # Model identifier (determines client routing)
  temperature: float = 0.2      # Sampling temperature (0.0 = greedy)
  max_output_tokens: int = 2048 # Maximum tokens in response
  log_calls: bool = false       # Enable structured call logging
  
  # Retry configuration
  max_retries: int = 5          # Maximum retry attempts on failure
  starting_delay: float = 1.0   # Initial retry delay (seconds)
  backoff_factor: float = 2.0   # Exponential backoff multiplier
  max_delay: float = 10.0       # Maximum retry delay (seconds)
```

### LLM Call Logging

When `log_calls: true`, all LLM calls are logged to structured JSON files:

**Directory structure**:
```
results_dir/YYYYMMDD_HHMMSS/llm_calls/
├── train/
│   ├── actions/           # Action generation calls
│   │   ├── action_episode_1_step_1_000001_TIMESTAMP.json
│   │   └── ...
│   └── reflections/       # Reflection calls (ReflexionAgent)
│       ├── reflection_episode_1_000001_TIMESTAMP.json
│       └── ...
└── validation/
    └── val_0/            # Organized by validation checkpoint
        └── actions/
            └── action_episode_1_step_1_000001_TIMESTAMP.json
```

**Log file format**:
```json
{
  "call_id": "uuid",
  "timestamp": "2024-01-15T10:30:45.123456Z",
  "mode": "train",
  "episode_index": 1,
  "step_index": 1,
  "call_type": "action",  // or "reflection", "feedback"
  "system_prompt": "You are a helpful assistant...",
  "user_prompt": "Solve this problem...",
  "response": "The answer is...",
  "metrics": {
    "duration": 1.234,
    "input_tokens": 512,
    "output_tokens": 128,
    "total_tokens": 640
  }
}
```

### Error Handling

All clients implement retry logic with exponential backoff:
1. Initial attempt
2. On failure: wait `starting_delay` seconds
3. Retry with delay multiplied by `backoff_factor`
4. Continue up to `max_retries` attempts
5. Cap delay at `max_delay`
6. Log detailed error information on final failure

**Common errors**:
- Rate limits → automatic retry with backoff
- Network timeouts → retry with timeout warnings
- API errors → logged with full context
- Truncation → warning logged, continues execution

### Usage in Agents

Agents receive an LM client during initialization:

```python
from src.lm.lm_factory import get_lm_client

# In agent __init__
self.lm = get_lm_client(config.lm_config, logger=logger)

# In agent act method
response = self.lm.call(system_prompt, user_prompt)
```

The runtime enables call logging when configured:
```python
# In main.py
enable_json_logging(results_dir / "llm_calls")
```


