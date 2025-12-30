"""
Layer capture mechanism for per-layer hidden states during forward pass.
Used for layer-wise distillation in cartridge training.
"""
from dataclasses import dataclass, field
from typing import Optional, Union

import torch


@dataclass
class LayerCapture:
    """
    Lightweight recorder for per-layer hidden states during forward pass.
    Used for layer-wise distillation: captures post-attention and post-MLP residuals.
    
    Usage:
        capture = LayerCapture(signals=["post_attn", "post_mlp"], layers="all")
        # Pass through model batch, after forward:
        # capture.captured["post_attn"] is dict[layer_idx -> Tensor]
    
    Signals:
        - "post_attn": hidden state after attention residual add
        - "post_mlp": hidden state after MLP residual add (layer output)
    """
    # Which signals to capture: "post_attn", "post_mlp"
    signals: list[str] = field(default_factory=list)
    # Which layers to capture: "all" or list of layer indices
    layers: Union[str, list[int]] = "all"
    # Optional: token indices to capture (if None, capture all tokens)
    token_indices: Optional[torch.Tensor] = None
    
    # Storage for captured tensors: signal_name -> {layer_idx -> Tensor}
    captured: dict[str, dict[int, torch.Tensor]] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.captured:
            self.captured = {sig: {} for sig in self.signals}
    
    def should_capture(self, signal: str, layer_idx: int) -> bool:
        """Check if we should capture this signal at this layer."""
        if signal not in self.signals:
            return False
        if self.layers == "all":
            return True
        return layer_idx in self.layers
    
    def record(self, signal: str, layer_idx: int, hidden_states: torch.Tensor):
        """Record hidden states for a given signal and layer."""
        if not self.should_capture(signal, layer_idx):
            return
        
        if self.token_indices is not None:
            # Only capture at specified token indices
            captured_states = hidden_states[:, self.token_indices, :]
        else:
            captured_states = hidden_states
        
        self.captured[signal][layer_idx] = captured_states
    
    def clear(self):
        """Clear all captured tensors."""
        self.captured = {sig: {} for sig in self.signals}
    
    def get_all_layers(self, signal: str) -> Optional[torch.Tensor]:
        """
        Stack all captured layers for a given signal into a single tensor.
        Returns: Tensor of shape [n_layers, batch, seq_len, hidden_dim] or None if empty.
        """
        if signal not in self.captured or not self.captured[signal]:
            return None
        
        layer_indices = sorted(self.captured[signal].keys())
        tensors = [self.captured[signal][idx] for idx in layer_indices]
        return torch.stack(tensors, dim=0)
