import abc
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
from typing import Optional, Literal

from pydrantic import ObjectConfig, BaseConfig
from pydantic import Field
import torch
import torch.nn as nn

from cartridges.utils import get_logger

logger = get_logger(__name__)

@dataclass
class AttnConfig:
    n_layers: int
    n_heads: int
    head_dim: int

CARTRIDGE_SEQ_ID = -1


class PositionalEmbeddingConfig(BaseConfig):
    _pass_as_config = True
    enabled: bool = False
    init_mode: Literal["zero", "random"] = "zero"
    random_std: float = 0.02
    zero_init_last_layer: bool = False


class CartridgeParametrization(nn.Module, abc.ABC):
    """
    Responsible for producing the cartridge (prefix) K/V tensors for all layers.

    It owns *all* trainable parameters associated with the cartridge, and knows
    how many frozen/trainable tokens there are.
    """

    def __init__(
        self,
        attn_config: AttnConfig,
        init_keys: list[torch.Tensor],
        init_values: list[torch.Tensor],
        num_frozen_tokens: int,
    ):
        super().__init__()
        self.attn_config = attn_config
        self.num_frozen_tokens = num_frozen_tokens
        self.num_init_tokens = init_keys[0].shape[2]
        self.num_trainable_tokens = self.num_init_tokens - num_frozen_tokens

    @abc.abstractmethod
    def get_frozen(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Returns lists of length n_layers of frozen keys/values,
        each of shape (1, n_heads, num_frozen_tokens, head_dim).
        Can return empty lists if there are no frozen tokens.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_trainable(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Returns lists of length n_layers of *current* trainable prefix keys/values,
        each of shape (1, n_heads, num_trainable_tokens, head_dim).

        This is where parametrization (offsets, MLP, etc.) is applied.
        """
        raise NotImplementedError

    def logging_metrics(self) -> dict[str, torch.Tensor]:
        """Optional hook for logging parametrization-specific metrics."""
        return {}


class OffsetParametrization(CartridgeParametrization):
    """Offset-based parametrization: trainable = reference + offset."""

    def __init__(
        self,
        attn_config: AttnConfig,
        init_keys: list[torch.Tensor],
        init_values: list[torch.Tensor],
        num_frozen_tokens: int,
    ):
        super().__init__(attn_config, init_keys, init_values, num_frozen_tokens)

        n_layers = attn_config.n_layers

        # Separate frozen vs reference
        if num_frozen_tokens > 0:
            self.frozen_keys = nn.ParameterList(
                [
                    nn.Parameter(k[:, :, :num_frozen_tokens].contiguous(), requires_grad=False)
                    for k in init_keys
                ]
            )
            self.frozen_values = nn.ParameterList(
                [
                    nn.Parameter(v[:, :, :num_frozen_tokens].contiguous(), requires_grad=False)
                    for v in init_values
                ]
            )
        else:
            self.frozen_keys = nn.ParameterList([])
            self.frozen_values = nn.ParameterList([])

        self.reference_keys = nn.ParameterList(
            [
                nn.Parameter(k[:, :, num_frozen_tokens:].contiguous(), requires_grad=False)
                for k in init_keys
            ]
        )
        self.reference_values = nn.ParameterList(
            [
                nn.Parameter(v[:, :, num_frozen_tokens:].contiguous(), requires_grad=False)
                for v in init_values
            ]
        )

        self.trainable_key_offsets = nn.ParameterList(
            [nn.Parameter(torch.zeros_like(ref_k)) for ref_k in self.reference_keys]
        )
        self.trainable_value_offsets = nn.ParameterList(
            [nn.Parameter(torch.zeros_like(ref_v)) for ref_v in self.reference_values]
        )

    def get_frozen(self):
        return list(self.frozen_keys), list(self.frozen_values)

    def get_trainable(self):
        keys = [
            ref_k + offset_k
            for ref_k, offset_k in zip(self.reference_keys, self.trainable_key_offsets)
        ]
        values = [
            ref_v + offset_v
            for ref_v, offset_v in zip(self.reference_values, self.trainable_value_offsets)
        ]
        return keys, values

    def logging_metrics(self) -> dict[str, torch.Tensor]:
        key_norms = torch.stack([p.norm() for p in self.trainable_key_offsets])
        value_norms = torch.stack([p.norm() for p in self.trainable_value_offsets])
        ref_key_norms = torch.stack([p.norm() for p in self.reference_keys])
        ref_value_norms = torch.stack([p.norm() for p in self.reference_values])
        return {
            "mean_key_offset_norm": key_norms.mean(),
            "mean_value_offset_norm": value_norms.mean(),
            "mean_reference_key_norm": ref_key_norms.mean(),
            "mean_reference_value_norm": ref_value_norms.mean(),
        }


class ResidualMLPParametrization(CartridgeParametrization):
    """MLP + residual parametrization: trainable = MLP(reference) + reference."""

    class Config(ObjectConfig):
        _pass_as_config = True
        hidden_multiplier: float = 4.0  # width of the MLP
        activation: Literal["relu", "gelu"] = "relu"
        share_across_layers: bool = True
        zero_init_last_layer: bool = False  # Zero-initialize last layer to keep init identical

    def __init__(
        self,
        config: Config,
        attn_config: AttnConfig,
        init_keys: list[torch.Tensor],
        init_values: list[torch.Tensor],
        num_frozen_tokens: int,
    ):
        super().__init__(attn_config, init_keys, init_values, num_frozen_tokens)
        self.cfg = config

        # Save reference embeddings like OffsetParametrization,
        # but now they are the *input* to the MLP.
        if num_frozen_tokens > 0:
            self.frozen_keys = nn.ParameterList(
                [
                    nn.Parameter(k[:, :, :num_frozen_tokens].contiguous(), requires_grad=False)
                    for k in init_keys
                ]
            )
            self.frozen_values = nn.ParameterList(
                [
                    nn.Parameter(v[:, :, :num_frozen_tokens].contiguous(), requires_grad=False)
                    for v in init_values
                ]
            )
        else:
            self.frozen_keys = nn.ParameterList([])
            self.frozen_values = nn.ParameterList([])

        self.reference_keys = nn.ParameterList(
            [
                nn.Parameter(k[:, :, num_frozen_tokens:].contiguous(), requires_grad=False)
                for k in init_keys
            ]
        )
        self.reference_values = nn.ParameterList(
            [
                nn.Parameter(v[:, :, num_frozen_tokens:].contiguous(), requires_grad=False)
                for v in init_values
            ]
        )

        head_dim = attn_config.head_dim
        hidden_dim = int(head_dim * self.cfg.hidden_multiplier)

        # Get dtype from input tensors to ensure MLP matches
        input_dtype = init_keys[0].dtype

        def make_mlp():
            act = nn.ReLU() if self.cfg.activation == "relu" else nn.GELU()
            last_layer = nn.Linear(hidden_dim, head_dim)
            
            # Zero-initialize last layer if requested (to keep init identical)
            if self.cfg.zero_init_last_layer:
                nn.init.zeros_(last_layer.weight)
                nn.init.zeros_(last_layer.bias)
            
            mlp = nn.Sequential(
                nn.Linear(head_dim, hidden_dim),
                act,
                last_layer,
            ).to(dtype=input_dtype)
            
            return mlp

        if self.cfg.share_across_layers:
            self.key_mlp = make_mlp()
            self.value_mlp = make_mlp()
        else:
            self.key_mlp = nn.ModuleList([make_mlp() for _ in range(attn_config.n_layers)])
            self.value_mlp = nn.ModuleList([make_mlp() for _ in range(attn_config.n_layers)])

    def _apply_mlp(self, mlp, x: torch.Tensor) -> torch.Tensor:
        # x: (1, n_heads, num_trainable_tokens, head_dim)
        b, h, t, d = x.shape
        y = x.view(-1, d)  # (b*h*t, d)
        y = mlp(y)
        return (y.view(b, h, t, d) + x)  # residual

    def get_frozen(self):
        return list(self.frozen_keys), list(self.frozen_values)

    def get_trainable(self):
        keys, values = [], []
        for layer_idx, (ref_k, ref_v) in enumerate(
            zip(self.reference_keys, self.reference_values)
        ):
            if isinstance(self.key_mlp, nn.ModuleList):
                k_mlp = self.key_mlp[layer_idx]
                v_mlp = self.value_mlp[layer_idx]
            else:
                k_mlp = self.key_mlp
                v_mlp = self.value_mlp

            keys.append(self._apply_mlp(k_mlp, ref_k))
            values.append(self._apply_mlp(v_mlp, ref_v))

        return keys, values

    def logging_metrics(self) -> dict[str, torch.Tensor]:
        # Example: norms of outputs vs references
        ks, vs = self.get_trainable()
        ref_k_norm = torch.stack([rk.norm() for rk in self.reference_keys]).mean()
        ref_v_norm = torch.stack([rv.norm() for rv in self.reference_values]).mean()
        k_norm = torch.stack([k.norm() for k in ks]).mean()
        v_norm = torch.stack([v.norm() for v in vs]).mean()
        return {
            "mean_reference_key_norm": ref_k_norm,
            "mean_reference_value_norm": ref_v_norm,
            "mean_mlp_key_norm": k_norm,
            "mean_mlp_value_norm": v_norm,
        }


class TrainableCache(nn.Module):
    """A trainable packed cache for generation with FlexAttention.
    
    The cache must do two things, which a standard Hugging Face cache does not:

    - Keep track of sequence membership of the cache and expose it to the model via
    the seq_ids method. The model will use this once per forward pass to construct 
    the appropriate block mask. 
    - Keep track of keys and values and expose them to the model in a packed manner via 
    the update method.
    
    TODO (Sabri): Ensure that tokens from the same sequence are contiguous. Eventually,
    should just page the keys and values.

    Args:
        config: The attention configuration, which we use to construct the 
        init_keys (list[torch.Tensor], optional): A `config.n_layers` length list of 
            trainable keys for the cache, should be of shape (1, n_heads, num_trainable_tokens, head_dim).
        init_values (list[torch.Tensor]): A `config.n_layers` length list of 
            trainable values for the cache, should be of shape (1, n_heads, num_trainable_tokens, head_dim).
        num_frozen_tokens (int): The number of the trainable tokens to freeze at the 
            beginning of the cache.
    """
    def __init__(
        self,        
        config: AttnConfig,
        init_keys: list[torch.Tensor]=None,
        init_values: list[torch.Tensor]=None,
        num_frozen_tokens: int = 0,
        parametrization: Optional[CartridgeParametrization] = None,
        positional_embeddings: Optional[PositionalEmbeddingConfig] = None,
    ):
        super().__init__()
        self.config = config
        self._keys = [None] * config.n_layers  # List of tensors per layer
        self._values = [None] * config.n_layers  # List of tensors per layer
        self._layer_seq_ids: list[Optional[torch.Tensor]] = [None] * config.n_layers
        self._num_tokens = 0
        self._positional_embeddings_cfg = positional_embeddings
        self._pos_emb_enabled = bool(
            positional_embeddings is not None and positional_embeddings.enabled
        )
        self._key_positional_embeddings: Optional[nn.ParameterList] = None
        self._value_positional_embeddings: Optional[nn.ParameterList] = None

        assert (init_keys is None) == (init_values is None)
        if init_keys is None:
            # No cartridge / no parametrization – pure runtime cache
            self._num_frozen_tokens = 0
            self._num_trainable_tokens = 0
            self.parametrization = None
            self._seq_ids = None
            self._init_seq_ids = None
            self._pos_emb_enabled = False
            return

        self._num_init_tokens = init_keys[0].shape[2]
        self._num_frozen_tokens = num_frozen_tokens
        self._num_trainable_tokens = self._num_init_tokens - num_frozen_tokens
        assert len(init_keys) == config.n_layers == len(init_values)
        
        # we initialize the seq ids for the first 
        # `num_trainable_tokens + num_frozen_tokens` tokens to -1, which means that 
        # the tokens are part of the cartridge and should be attended to by 
        # all tokens.
        _seq_ids =torch.full(
            (self._num_init_tokens,),
            fill_value=CARTRIDGE_SEQ_ID, 
            dtype=torch.long,
        )
        self.register_buffer("_init_seq_ids", _seq_ids)
        self.register_buffer("_seq_ids", _seq_ids)  # .to moves the tensor to the correct device

        for vec in itertools.chain(init_keys, init_values):
            assert vec.shape == (1, config.n_heads, self._num_init_tokens, config.head_dim)

        if self._pos_emb_enabled:
            self._init_positional_embeddings(dtype=init_keys[0].dtype)

        # Plug in parametrization (default: OffsetParametrization)
        if parametrization is None:
            self.parametrization = OffsetParametrization(
                attn_config=config,
                init_keys=init_keys,
                init_values=init_values,
                num_frozen_tokens=num_frozen_tokens,
            )
        else:
            self.parametrization = parametrization

        # Register parametrization as submodule
        if self.parametrization is not None:
            self.add_module("parametrization", self.parametrization)

        # Convenience cached numbers
        assert (
            self.parametrization.num_trainable_tokens == self._num_trainable_tokens
        )
        assert self.parametrization.num_frozen_tokens == self._num_frozen_tokens

        logger.info(f"num_trainable_tokens: {self._num_trainable_tokens}")
        logger.info(f"num_frozen_tokens: {self._num_frozen_tokens}")

    def _init_positional_embeddings(self, dtype: torch.dtype):
        assert self._positional_embeddings_cfg is not None
        shape = (1, self.config.n_heads, 1, self.config.head_dim)
        init_mode = self._positional_embeddings_cfg.init_mode
        std = self._positional_embeddings_cfg.random_std

        def _make_param():
            if init_mode == "random":
                tensor = torch.randn(shape, dtype=dtype) * std
            else:
                tensor = torch.zeros(shape, dtype=dtype)
            return nn.Parameter(tensor)

        self._key_positional_embeddings = nn.ParameterList(
            [_make_param() for _ in range(self.config.n_layers)]
        )
        self._value_positional_embeddings = nn.ParameterList(
            [_make_param() for _ in range(self.config.n_layers)]
        )

    def _apply_positional_embedding(
        self, tensor: torch.Tensor, layer_idx: int, is_key: bool
    ) -> torch.Tensor:
        if not self._pos_emb_enabled:
            return tensor
        embeddings = (
            self._key_positional_embeddings if is_key else self._value_positional_embeddings
        )
        if embeddings is None or len(embeddings) == 0:
            return tensor
        return tensor + embeddings[layer_idx]
    
    @property
    def trainable_keys(self) -> list[torch.Tensor]:
        if self.parametrization is None:
            return []
        ks, _ = self.parametrization.get_trainable()
        return [
            self._apply_positional_embedding(k, idx, is_key=True)
            for idx, k in enumerate(ks)
        ]
    
    @property
    def trainable_values(self) -> list[torch.Tensor]:
        if self.parametrization is None:
            return []
        _, vs = self.parametrization.get_trainable()
        return [
            self._apply_positional_embedding(v, idx, is_key=False)
            for idx, v in enumerate(vs)
        ]

    @property
    def frozen_keys(self) -> list[torch.Tensor]:
        if self.parametrization is None:
            return []
        ks, _ = self.parametrization.get_frozen()
        return [
            self._apply_positional_embedding(k, idx, is_key=True)
            for idx, k in enumerate(ks)
        ]

    @property
    def frozen_values(self) -> list[torch.Tensor]:
        if self.parametrization is None:
            return []
        _, vs = self.parametrization.get_frozen()
        return [
            self._apply_positional_embedding(v, idx, is_key=False)
            for idx, v in enumerate(vs)
        ]
    
    def positional_embedding_metrics(self) -> dict[str, torch.Tensor]:
        """Returns metrics for positional embeddings (norms, etc.) for logging."""
        if not self._pos_emb_enabled:
            return {}
        
        if self._key_positional_embeddings is None or len(self._key_positional_embeddings) == 0:
            return {}
        
        key_norms = torch.stack([p.norm() for p in self._key_positional_embeddings])
        value_norms = torch.stack([p.norm() for p in self._value_positional_embeddings])
        
        return {
            "mean_key_pos_embed_norm": key_norms.mean(),
            "mean_value_pos_embed_norm": value_norms.mean(),
            "max_key_pos_embed_norm": key_norms.max(),
            "max_value_pos_embed_norm": value_norms.max(),
            "min_key_pos_embed_norm": key_norms.min(),
            "min_value_pos_embed_norm": value_norms.min(),
        }
                
    def update(
        self, 
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
        new_seq_ids: torch.Tensor,
        layer_idx: int,
        skip_append: bool = False,
        return_seq_ids: bool = False,
    ):
        """Update the cache with new keys and values while maintaining sequence contiguity.
        
        Args:
            new_keys: (1, num_heads, seq_len, head_dim) tensor of new keys
            new_values: (1, num_heads, seq_len, head_dim) tensor of new values  
            new_seq_ids: (seq_len,) tensor of sequence ids for the new tokens
            layer_idx: index of the layer in the model.
            skip_append: if True, do not append the new keys and values to the cache, 
                just return the concatenation of the new_keys and values. 
            return_seq_ids: if True, also return the sequence ids corresponding to the
                concatenated keys/values that will be used by the attention module.
        """
        assert new_seq_ids.shape[0] == new_keys.shape[2]
        assert new_seq_ids.shape[0] == new_values.shape[2]

        if layer_idx == 0 and not skip_append:
            # we assume the same seq ids at every layer. This allows us to create
            # a single block mask for the entire model. 
            if self._seq_ids is None:
                self._seq_ids = new_seq_ids
            else:
                self._seq_ids = torch.cat([self._seq_ids, new_seq_ids], dim=0)
            self._num_tokens += new_keys.shape[2]
        
        keys = [new_keys]
        values = [new_values]

        if self._keys[layer_idx] is not None:
            # Concatenate along sequence dimension while maintaining contiguous sequences
            keys = [self._keys[layer_idx]] + keys
            values = [self._values[layer_idx]] + values

        layer_seq_ids_updated = False
        if not skip_append:
            self._keys[layer_idx] = torch.cat(keys, dim=2)
            self._values[layer_idx] = torch.cat(values, dim=2)
            if self._layer_seq_ids[layer_idx] is None:
                self._layer_seq_ids[layer_idx] = new_seq_ids
            else:
                self._layer_seq_ids[layer_idx] = torch.cat(
                    [self._layer_seq_ids[layer_idx], new_seq_ids], dim=0
                )
            layer_seq_ids_updated = True
        
        if self._num_trainable_tokens > 0 and self.parametrization is not None:
            trainable_keys, trainable_values = self.parametrization.get_trainable()
            keys = [
                self._apply_positional_embedding(trainable_keys[layer_idx], layer_idx, is_key=True)
            ] + keys
            values = [
                self._apply_positional_embedding(trainable_values[layer_idx], layer_idx, is_key=False)
            ] + values
        
        if self._num_frozen_tokens > 0 and self.parametrization is not None:
            frozen_keys, frozen_values = self.parametrization.get_frozen()
            keys = [
                self._apply_positional_embedding(frozen_keys[layer_idx], layer_idx, is_key=True)
            ] + keys
            values = [
                self._apply_positional_embedding(frozen_values[layer_idx], layer_idx, is_key=False)
            ] + values
        
        # BB: TODO: why is this here?
        # if self._num_trainable_tokens == 0 and self._num_frozen_tokens == 0:
        #     return self._keys[layer_idx], self._values[layer_idx]

        concatenated_keys = torch.cat(keys, dim=2)
        concatenated_values = torch.cat(values, dim=2)

        if return_seq_ids:
            device = new_seq_ids.device
            seq_components: list[torch.Tensor] = []

            if self._num_frozen_tokens > 0 and self._init_seq_ids is not None:
                seq_components.append(
                    self._init_seq_ids[: self._num_frozen_tokens].to(device)
                )
            if self._num_trainable_tokens > 0 and self._init_seq_ids is not None:
                start = self._num_frozen_tokens
                seq_components.append(
                    self._init_seq_ids[start : start + self._num_trainable_tokens].to(
                        device
                    )
                )

            if self._layer_seq_ids[layer_idx] is not None:
                seq_components.append(self._layer_seq_ids[layer_idx].to(device))
                if not layer_seq_ids_updated:
                    seq_components.append(new_seq_ids)
            else:
                seq_components.append(new_seq_ids)

            concatenated_seq_ids = torch.cat(seq_components, dim=0)
            return concatenated_keys, concatenated_values, concatenated_seq_ids

        return concatenated_keys, concatenated_values
    
    def num_tokens(self) -> int:
        """Get the sequence length of the cache."""
        return self._num_frozen_tokens + self._num_trainable_tokens + self._num_tokens
    
    def num_cartridge_tokens(self) -> int:
        """Get the number of tokens in the cartridge."""
        return self._num_frozen_tokens + self._num_trainable_tokens
    
    def seq_ids(self) -> Optional[torch.Tensor]:
        """Returns the sequence ids of the cache."""
        return self._seq_ids
       
    def clear(self):
        self._keys = [None] * self.config.n_layers
        self._values = [None] * self.config.n_layers
        self._layer_seq_ids = [None] * self.config.n_layers
        self._num_tokens = 0
        self._seq_ids = self._init_seq_ids

    def save(self, path: str):
        """Saves the trainable keys and values to the specified path."""
        trainable_keys, trainable_values = (
            self.parametrization.get_trainable() if self.parametrization else ([], [])
        )
        frozen_keys, frozen_values = (
            self.parametrization.get_frozen() if self.parametrization else ([], [])
        )

        if self.parametrization is not None and self._pos_emb_enabled:
            trainable_keys = [
                self._apply_positional_embedding(k, idx, is_key=True)
                for idx, k in enumerate(trainable_keys)
            ]
            trainable_values = [
                self._apply_positional_embedding(v, idx, is_key=False)
                for idx, v in enumerate(trainable_values)
            ]
            frozen_keys = [
                self._apply_positional_embedding(k, idx, is_key=True)
                for idx, k in enumerate(frozen_keys)
            ]
            frozen_values = [
                self._apply_positional_embedding(v, idx, is_key=False)
                for idx, v in enumerate(frozen_values)
            ]
        torch.save(
            {
                "trainable_keys": trainable_keys,
                "trainable_values": trainable_values,
                "frozen_keys": frozen_keys,
                "frozen_values": frozen_values,
            },
            path,
        )

    @classmethod
    def from_pretrained(cls, path: str, device: Optional[str] = None):
        if not isinstance(path, str):
            raise TypeError(f"path must be a string, got {type(path)}")
        print(path)
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Ensure necessary keys are in the checkpoint
        for key in ["trainable_keys", "trainable_values", "frozen_keys", "frozen_values"]:
            if key not in checkpoint:
                raise KeyError(f"Key '{key}' not found in checkpoint")

        n_layers = len(checkpoint["trainable_keys"])
        n_heads = checkpoint["trainable_keys"][0].size(1)
        num_tokens = checkpoint["trainable_keys"][0].size(2)
        head_dim = checkpoint["trainable_keys"][0].size(3)

        # Allow empty frozen_keys list (when num_frozen_tokens == 0)
        if checkpoint["frozen_keys"] and len(checkpoint["frozen_keys"]) != n_layers:
            raise AssertionError(
                "Mismatch in number of layers between trainable and fixed keys"
            )
        if checkpoint["frozen_keys"]:
            if (
                checkpoint["frozen_keys"][0].size(1) != n_heads
                or checkpoint["frozen_keys"][0].size(3) != head_dim
            ):
                raise AssertionError(
                    "Mismatch in head configuration between trainable and fixed keys"
                )

        config = AttnConfig(n_layers=n_layers, n_heads=n_heads, head_dim=head_dim)
        # Here, num_tokens is inferred from trainable keys, but note that the total tokens may be different if fixed tokens exist.
        # The number of fixed tokens can be inferred from frozen_keys if available.
        num_frozen_tokens = (
            checkpoint["frozen_keys"][0].size(2) if checkpoint["frozen_keys"] else 0
        )

        # Reconstruct init_keys and init_values from checkpoint
        if num_frozen_tokens > 0:
            init_keys = [
                torch.cat([fixed, trainable], dim=2).contiguous()
                for fixed, trainable in zip(
                    checkpoint["frozen_keys"], checkpoint["trainable_keys"]
                )
            ]
            init_values = [
                torch.cat([fixed, trainable], dim=2).contiguous()
                for fixed, trainable in zip(
                    checkpoint["frozen_values"], checkpoint["trainable_values"]
                )
            ]
        else:
            # No frozen tokens, just use trainable keys/values
            init_keys = list(checkpoint["trainable_keys"])
            init_values = list(checkpoint["trainable_values"])

        # Create OffsetParametrization from loaded data (backward compatible)
        parametrization = OffsetParametrization(
            attn_config=config,
            init_keys=init_keys,
            init_values=init_values,
            num_frozen_tokens=num_frozen_tokens,
        )

        return cls(
            config=config,
            init_keys=init_keys,
            init_values=init_values,
            num_frozen_tokens=num_frozen_tokens,
            parametrization=parametrization,
        )


# Parametrization registry
PARAM_REGISTRY = {
    "offset": OffsetParametrization,
    "mlp_residual": ResidualMLPParametrization,
}


def create_parametrization(
    parametrization_type: str,
    parametrization_config: dict,
    attn_config: AttnConfig,
    init_keys: list[torch.Tensor],
    init_values: list[torch.Tensor],
    num_frozen_tokens: int,
) -> CartridgeParametrization:
    """Helper function to create a parametrization from config."""
    ParamCls = PARAM_REGISTRY[parametrization_type]

    if ParamCls is ResidualMLPParametrization:
        param_cfg = ResidualMLPParametrization.Config(**parametrization_config)
        parametrization = ParamCls(
            config=param_cfg,
            attn_config=attn_config,
            init_keys=init_keys,
            init_values=init_values,
            num_frozen_tokens=num_frozen_tokens,
        )
    else:
        parametrization = ParamCls(
            attn_config=attn_config,
            init_keys=init_keys,
            init_values=init_values,
            num_frozen_tokens=num_frozen_tokens,
        )

    return parametrization


class KVCacheFactory(abc.ABC):
    class Config(ObjectConfig):
        _pass_as_config = True

        # SE (03/26): we freeze the first token to prevent forgetting
        num_frozen_tokens: int = 1

        # Parametrization configuration
        parametrization_type: Literal["offset", "mlp_residual"] = "offset"
        parametrization_config: dict = Field(default_factory=dict)
        positional_embeddings: PositionalEmbeddingConfig = Field(default_factory=PositionalEmbeddingConfig)

    def __init__(self, config: Config):
        self.config = config

    @abc.abstractmethod
    def initialize_kv_cache(
        self, tokenizer, model, attn_config: AttnConfig 
    ) -> TrainableCache:
        raise NotImplementedError()


class KVCacheFactoryWithStateSaving(abc.ABC):
    class Config(KVCacheFactory.Config):
        directory: str
        is_wandb: bool
        force_recreate: bool = False

    def __init__(self, config: Config):
        self.config = config

    @abc.abstractmethod
    def initalize_kv_cache_impl(
        self,
        tokenizer,
        model,
        attn_config: AttnConfig,
    ) -> tuple[TrainableCache, dict]:
        raise NotImplementedError()

    @property
    def local_kv_cache_path(self) -> Path:
        # TODO: better file extension
        return Path(self.config.directory) / "kv_cache.torch"

    @property
    def local_metadata_path(self) -> Path:
        # TODO: better file extension
        return Path(self.config.directory) / "metadata.json"

    def maybe_load_cached(self) -> Optional[TrainableCache]:
        if self.config.force_recreate:
            return

        if not self.config.is_wandb:
            if self.local_kv_cache_path.exists():
                logger.info(
                    f"State Saving KV initializer: loading KV cache from: {self.local_kv_cache_path}"
                )
                return TrainableCache.from_pretrained(
                    str(self.local_kv_cache_path.absolute()),
                )

            return

        raise NotImplementedError("Need to add saving to wanb")

    def initalize_kv_cache(
        self, tokenizer, model, attn_config: AttnConfig
    ) -> TrainableCache:
        maybe_cache = self.maybe_load_cached()
        if maybe_cache is not None:
            assert (
                maybe_cache._num_trainable_tokens + maybe_cache._num_frozen_tokens
                == self.config.num_tokens
            )
            assert maybe_cache.config == attn_config
            return maybe_cache

        cache, metadata = self.initalize_kv_cache_impl(
            tokenizer, model, attn_config
        )

        Path(self.config.directory).mkdir(parents=True, exist_ok=True)

        with open(self.local_metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        cache.save(str(self.local_kv_cache_path.absolute()))
        logger.info(
            f"State Saving KV initializer: saving KV cache to: {self.local_kv_cache_path}"
        )

        return cache
