import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from logging import Logger

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.lm.language_model import LMConfig, LanguageModel
from src.utils import logger as jsonlogger

# Set up cartridges environment variables BEFORE importing cartridges modules
# File is at: src/lm/hf_client.py
# So REPO_ROOT is: parent.parent.parent
REPO_ROOT = Path(__file__).parent.parent.parent
CARTRIDGES_DIR = REPO_ROOT / "third_party" / "cartridges"

# Cartridges requires these environment variables
os.environ["CARTRIDGES_DIR"] = str(CARTRIDGES_DIR)
if "CARTRIDGES_OUTPUT_DIR" not in os.environ:
    os.environ["CARTRIDGES_OUTPUT_DIR"] = str(REPO_ROOT / "outputs")

# Add third_party/cartridges to path so we can import cartridges modules
sys.path.insert(0, str(CARTRIDGES_DIR))

# Now we can import cartridges modules
from cartridges.generation import flex_generate
from cartridges.cache import TrainableCache, AttnConfig
from cartridges.datasets import LLAMA_CARTRIDGE_TEMPLATE


class GenerationTruncatedError(RuntimeError):
    """Raised when the model likely hit the max output tokens and truncated."""


class HFClientConfig(LMConfig):
    model: str  # Model identifier (e.g., "meta-llama/Llama-3.1-8B-Instruct")
    cartridges: Optional[List[Dict[str, Any]]] = None  # Cartridge configs
    cartridge_dir: Optional[str] = None  # Base directory for loading cartridges
    device: str = "cuda"  # Device for model and cache
    torch_dtype: str = "bfloat16"  # Model dtype
    use_unrotated_queries_for_cartridges: bool = False  # Cartridge query rotation setting
    non_cartridge_start_position_id_offset: int = 0  # Position offset for non-cartridge tokens
    load_kwargs: Optional[Dict[str, Any]] = None  # Additional kwargs for from_pretrained
    stop_sequences: Optional[List[str]] = None  # Stop sequences (converted to token IDs)
    model_cls: Optional[str] = None  # Optional: "FlexLlamaForCausalLM" or "FlexQwen3ForCausalLM" (auto-detected if None)


class HFClient(LanguageModel):
    """HuggingFace client for local model inference with cartridges.
    
    Uses flex_generate from cartridges for generation with TrainableCache support.
    """

    def __init__(self, config: HFClientConfig, logger: Optional[Logger] = None):
        super().__init__(config=config, logger=logger)
        self._model: Optional[torch.nn.Module] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._cache: Optional[TrainableCache] = None
        
        # Load model and tokenizer
        self._load_model()
        self._load_tokenizer()
        
        # Load cache if configured (uses first cartridge from config, matching generate_with_hf pattern)
        if self.cfg.cartridges and self.cfg.cartridge_dir:
            self._load_cache()
        
        # Convert stop sequences to token IDs if provided
        # Tokenizer is guaranteed to be loaded at this point
        assert self._tokenizer is not None, "Tokenizer must be loaded"
        self._stop_token_ids: Optional[List[int]] = None
        if self.cfg.stop_sequences:
            self._stop_token_ids = []
            for seq in self.cfg.stop_sequences:
                token_ids = self._tokenizer.encode(seq, add_special_tokens=False)
                self._stop_token_ids.extend(token_ids)
            # Also add EOS token if available
            if self._tokenizer.eos_token_id is not None:
                if self._tokenizer.eos_token_id not in self._stop_token_ids:
                    self._stop_token_ids.append(self._tokenizer.eos_token_id)
        elif self._tokenizer.eos_token_id is not None:
            self._stop_token_ids = [self._tokenizer.eos_token_id]

    @property
    def cfg(self) -> HFClientConfig:
        return self.config  # type: ignore[return-value]

    def _load_model(self) -> None:
        """Load the model from HuggingFace.
        
        Uses Flex model classes (FlexLlamaForCausalLM, FlexQwen3ForCausalLM) which support
        cartridge-specific config parameters. Auto-detects model type if model_cls not specified.
        
        All hyperparameters are passed via load_kwargs to from_pretrained(),
        which forwards config parameters to the model's config class.
        """
        model_id = self.cfg.model
        # Start with user-provided load_kwargs or empty dict
        load_kwargs = dict(self.cfg.load_kwargs) if self.cfg.load_kwargs else {}
        
        # Determine torch dtype
        if self.cfg.torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif self.cfg.torch_dtype == "float16":
            torch_dtype = torch.float16
        elif self.cfg.torch_dtype == "float32":
            torch_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported torch_dtype: {self.cfg.torch_dtype}")
        
        # Add model loading parameters to load_kwargs if not already specified
        # These are passed directly to from_pretrained()
        if "torch_dtype" not in load_kwargs:
            load_kwargs["torch_dtype"] = torch_dtype
        
        # Add model config parameters to load_kwargs if not already specified
        # These get passed to the model's config class (e.g., LlamaConfig) during initialization
        if "non_cartridge_start_position_id_offset" not in load_kwargs:
            load_kwargs["non_cartridge_start_position_id_offset"] = self.cfg.non_cartridge_start_position_id_offset
        if "use_unrotated_queries_for_cartridges" not in load_kwargs:
            load_kwargs["use_unrotated_queries_for_cartridges"] = self.cfg.use_unrotated_queries_for_cartridges
        
        # Determine which model class to use
        # Flex models are required for cartridge-specific parameters to work
        # Auto-detect based on model name
        model_id_lower = model_id.lower()
        if "qwen" in model_id_lower:
            from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
            model_cls = FlexQwen3ForCausalLM
            self.logger.info(f"Auto-detected Qwen model, using FlexQwen3ForCausalLM")
        elif "llama" in model_id_lower:
            from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
            model_cls = FlexLlamaForCausalLM
            self.logger.info(f"Auto-detected Llama model, using FlexLlamaForCausalLM")
        else:
            raise ValueError(f"Model {model_id} not supported")
        
        self.logger.info(f"Loading model: {model_id} using {model_cls.__name__}")
        self.logger.debug(f"Model config parameters: non_cartridge_start_position_id_offset={load_kwargs.get('non_cartridge_start_position_id_offset')}, use_unrotated_queries_for_cartridges={load_kwargs.get('use_unrotated_queries_for_cartridges')}")
        
        self._model = model_cls.from_pretrained(
            model_id,
            **load_kwargs
        )
        
        # Move model to device if not using device_map
        # device_map handles device placement automatically, so we skip manual .to() if it's set
        if "device_map" not in load_kwargs:
            device = torch.device(self.cfg.device)
            self._model = self._model.to(device)
        
        self._model.eval()
        self.logger.info(f"Model loaded on {self.cfg.device}")

    def _load_tokenizer(self) -> None:
        """Load the tokenizer from HuggingFace."""
        model_id = self.cfg.model
        self.logger.info(f"Loading tokenizer: {model_id}")
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.logger.info("Tokenizer loaded")

    def _load_cache(self) -> None:
        """Load cache from local directory based on config.
        
        Matches generate_with_hf pattern: uses a single TrainableCache object.
        If multiple cartridges are configured, only the first one is loaded.
        """
        if not self.cfg.cartridges or not self.cfg.cartridge_dir:
            return
        
        if len(self.cfg.cartridges) == 0:
            self.logger.warning("Cartridges config is empty, no cache will be loaded")
            return
        
        # Use first cartridge (matching generate_with_hf pattern which takes a single cache)
        cartridge_cfg = self.cfg.cartridges[0]
        if len(self.cfg.cartridges) > 1:
            self.logger.warning(
                f"Multiple cartridges configured ({len(self.cfg.cartridges)}), "
                f"but only the first one will be used: {cartridge_cfg.get('id', 'unknown')}"
            )
        
        cartridge_id = cartridge_cfg.get("id")
        if not cartridge_id:
            raise ValueError(f"Cartridge config missing 'id': {cartridge_cfg}")
        
        resolved_device = torch.device(self.cfg.device)
        base_path = Path(self.cfg.cartridge_dir)
        
        # Cartridges are stored at: {cartridge_dir}/{cartridge_id}/cartridge.pt
        cartridge_path = base_path / cartridge_id / "cartridge.pt"
        
        if not cartridge_path.exists():
            raise FileNotFoundError(
                f"Cartridge file not found for ID '{cartridge_id}'. "
                f"Expected path: {cartridge_path}"
            )
        
        self.logger.info(f"Loading cache from cartridge: {cartridge_id}")
        cache = TrainableCache.from_pretrained(
            str(cartridge_path),
            device=resolved_device.type if resolved_device.type != "cuda" else None,
        )
        cache = cache.to(device=resolved_device)
        cache.eval()
        self._cache = cache
        self.logger.info(f"Cache loaded from cartridge {cartridge_id}")

    def call(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Call the language model with messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
        
        Returns:
            Dictionary containing 'text' and optionally 'metrics'
        """
        call_id = self._begin_call(messages)
        start_time = time.time()
        last_err: Optional[Exception] = None
        ctx: Dict[str, Any] = jsonlogger.json_get_context()
        mode = ctx.get("mode")
        should_retry_truncation = not (mode == "val" or mode == "validation")
        
        for attempt in range(1, self.config.max_retries + 2):
            try:
                # Ensure model and tokenizer are loaded
                assert self._model is not None, "Model must be loaded"
                assert self._tokenizer is not None, "Tokenizer must be loaded"
                
                # Determine temperature based on mode
                temperature = (
                    self.cfg.val_temperature 
                    if (mode == "val" or mode == "validation") 
                    else self.cfg.train_temperature
                )
                
                # Convert messages to input format
                # Use LLAMA_CARTRIDGE_TEMPLATE when cache is present and model is Llama
                # This matches the template used in train.py's generate_with_hf and tokasaurus server
                if self._cache is not None and "llama" in self.cfg.model.lower():
                    # Use cartridge template when cache is present (matching train.py and tokasaurus)
                    input_ids = self._tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        chat_template=LLAMA_CARTRIDGE_TEMPLATE,
                        add_special_tokens=False,  # Important: matches dataset usage
                    )
                else:
                    # Use default template when no cache
                    input_ids = self._tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    )
                
                # Move to device and flatten
                input_ids = input_ids.to(self.cfg.device)
                flat_input_ids = input_ids.flatten()
                
                # Create seq_ids and position_ids for single sequence
                seq_ids = torch.zeros(
                    flat_input_ids.shape[0], 
                    dtype=torch.long, 
                    device=self.cfg.device
                )
                position_ids = torch.arange(
                    flat_input_ids.shape[0], 
                    dtype=torch.long, 
                    device=self.cfg.device
                )
                
                # Generate (cache is a single TrainableCache or None, matching generate_with_hf pattern)
                generated_tokens = flex_generate(
                    model=self._model,
                    tokenizer=self._tokenizer,
                    input_ids=flat_input_ids,
                    seq_ids=seq_ids,
                    position_ids=position_ids,
                    cache=self._cache,
                    stop_token_ids=self._stop_token_ids,
                    max_new_tokens=self.cfg.max_output_tokens,
                    temperature=temperature,
                    show_progress=False,
                )
                
                # Decode the response
                if 0 in generated_tokens and generated_tokens[0]:
                    text = self._tokenizer.decode(
                        generated_tokens[0], 
                        skip_special_tokens=True
                    )
                else:
                    text = ""
                
                # Check for truncation
                if len(generated_tokens.get(0, [])) >= self.cfg.max_output_tokens:
                    raise GenerationTruncatedError(
                        f"Generation likely truncated at max_output_tokens={self.cfg.max_output_tokens}"
                    )
                
                # Calculate metrics
                duration = time.time() - start_time
                input_token_count = len(flat_input_ids)
                output_token_count = len(generated_tokens.get(0, []))
                
                metrics: Dict[str, Any] = {
                    "duration": duration,
                    "input_tokens": input_token_count,
                    "thinking_tokens": None,
                    "output_tokens": output_token_count,
                    "total_tokens": input_token_count + output_token_count,
                    "temperature": temperature,
                }
                
                self._end_call(call_id, text, extra={"metrics": metrics})
                return {
                    "text": text,
                    "metrics": metrics,
                }
                
            except Exception as e:
                last_err = e
                error_type = type(e).__name__
                
                # Build detailed error context
                error_context = {
                    "model": self.cfg.model,
                    "error_type": error_type,
                    "messages_count": len(messages),
                }
                
                # Only retry on truncation during training
                if not (should_retry_truncation and isinstance(e, GenerationTruncatedError)):
                    self.logger.error(
                        f"HF call FAILED (no retry). Model: {self.cfg.model}, Error: {error_type}: {e}",
                        extra=error_context
                    )
                    break
                
                if attempt > self.config.max_retries:
                    self.logger.error(
                        f"HF call FAILED after {attempt} attempts. "
                        f"Model: {self.cfg.model}, Error: {error_type}: {e}",
                        extra=error_context
                    )
                    break
                
                delay = min(
                    self.config.starting_delay * (self.config.backoff_factor ** attempt),
                    self.config.max_delay,
                )
                self.logger.warning(
                    f"HF call failed at attempt {attempt}/{self.config.max_retries + 1}. "
                    f"Model: {self.cfg.model}, Error: {error_type}: {e}. "
                    f"Retrying in {delay:.2f}s...",
                    extra=error_context
                )
                time.sleep(delay)
        
        self._end_call(call_id, "", extra={"error": str(last_err) if last_err else "unknown"})
        return {"text": ""}

