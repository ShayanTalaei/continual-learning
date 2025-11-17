Here’s a concrete refactor plan that lets you plug in any cartridge / prefix parametrization (offsets, MLP+residual, etc.) while keeping the rest of the codebase essentially unchanged.

I’ll structure it as:

What TrainableCache is doing today

New abstraction: CartridgeParametrization

Refactor TrainableCache to use the abstraction

Implement two parametrizations: Offset (current behavior) and MLP+residual

Config / factory wiring

Saving / loading & “compile-to-plain-prefix” for deployment

Minimal changes needed outside cache.py

1. What TrainableCache does today

Right now TrainableCache is responsible for two orthogonal things:

Runtime KV cache for packed generation

Maintains _keys, _values, _layer_seq_ids, _seq_ids, _num_tokens

Exposes update(...), num_tokens, num_cartridge_tokens, seq_ids, clear

This is what the Flex model actually needs as past_key_values.

Prefix / cartridge parametrization

Stores:

frozen_keys / frozen_values: first num_frozen_tokens, never updated.

reference_keys / reference_values: base “reference” prefix tokens.

trainable_key_offsets / trainable_value_offsets: trainable params.

trainable_keys / trainable_values are reference + offset (current parametrization).

save/from_pretrained assume the above structure and save the materialized keys/values.

These two concerns are currently mixed together in TrainableCache. To support other parametrizations (MLP, low-rank adapters, etc.) cleanly, we should separate “how prefix KV is parametrized” from “how KV cache is used at runtime”.

2. New abstraction: CartridgeParametrization

Introduce a small, focused abstraction that only handles “how to produce prefix K/V tensors from parameters”.

# cache.py

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

    def num_frozen_tokens(self) -> int:
        return self.num_frozen_tokens

    def num_trainable_tokens(self) -> int:
        return self.num_trainable_tokens

    # Optional hook for logging
    def logging_metrics(self) -> dict[str, torch.Tensor]:
        return {}


Key points:

All parametrization-specific state (reference K/V, offsets, MLP weights, etc.) lives inside a CartridgeParametrization subclass.

Its public API is just: “give me frozen K/V” and “give me trainable K/V”.

3. Refactor TrainableCache to use the parametrization

Change TrainableCache so that:

It owns a CartridgeParametrization instance instead of raw frozen_keys, reference_keys, trainable_key_offsets, etc.

All KV concatenation logic in update consults self.parametrization rather than hard-coding offsets.

3.1. Constructor

Replace the current “reference + offset” logic with:

class TrainableCache(nn.Module):
    def __init__(
        self,
        config: AttnConfig,
        init_keys: list[torch.Tensor] | None = None,
        init_values: list[torch.Tensor] | None = None,
        num_frozen_tokens: int = 0,
        parametrization: CartridgeParametrization | None = None,
    ):
        super().__init__()
        self.config = config
        self._keys = [None] * config.n_layers
        self._values = [None] * config.n_layers
        self._layer_seq_ids: list[Optional[torch.Tensor]] = [None] * config.n_layers
        self._num_tokens = 0

        assert (init_keys is None) == (init_values is None)

        if init_keys is None:
            # No cartridge / no parametrization – pure runtime cache
            self._num_frozen_tokens = 0
            self._num_trainable_tokens = 0
            self.parametrization = None
            self._seq_ids = None
            self._init_seq_ids = None
            return

        self._num_init_tokens = init_keys[0].shape[2]
        self._num_frozen_tokens = num_frozen_tokens
        self._num_trainable_tokens = self._num_init_tokens - num_frozen_tokens

        # seq_ids for cartridge tokens (all -1)
        _seq_ids = torch.full(
            (self._num_init_tokens,),
            fill_value=CARTRIDGE_SEQ_ID,
            dtype=torch.long,
        )
        self.register_buffer("_init_seq_ids", _seq_ids)
        self.register_buffer("_seq_ids", _seq_ids)

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

        # Convenience cached numbers
        assert (
            self.parametrization.num_trainable_tokens() == self._num_trainable_tokens
        )
        assert self.parametrization.num_frozen_tokens() == self._num_frozen_tokens

3.2. Accessors

Replace the old trainable_keys/trainable_values properties with thin wrappers:

    @property
    def trainable_keys(self) -> list[torch.Tensor]:
        if self.parametrization is None:
            return []
        ks, _ = self.parametrization.get_trainable()
        return ks

    @property
    def trainable_values(self) -> list[torch.Tensor]:
        if self.parametrization is None:
            return []
        _, vs = self.parametrization.get_trainable()
        return vs

    @property
    def frozen_keys(self) -> list[torch.Tensor]:
        if self.parametrization is None:
            return []
        ks, _ = self.parametrization.get_frozen()
        return ks

    @property
    def frozen_values(self) -> list[torch.Tensor]:
        if self.parametrization is None:
            return []
        _, vs = self.parametrization.get_frozen()
        return vs


(If you want to keep frozen_keys/frozen_values as nn.ParameterList, you can expose them via properties instead of direct attributes.)

3.3. update method

Modify the prefix concatenation to go through the parametrization:

    def update(...):
        ...
        # After we update self._keys[layer_idx] / self._values[layer_idx]
        # and build `keys` / `values` lists for this layer:

        if self._num_trainable_tokens > 0 and self.parametrization is not None:
            trainable_keys, trainable_values = self.parametrization.get_trainable()
            keys = [trainable_keys[layer_idx]] + keys
            values = [trainable_values[layer_idx]] + values

        if self._num_frozen_tokens > 0 and self.parametrization is not None:
            frozen_keys, frozen_values = self.parametrization.get_frozen()
            keys = [frozen_keys[layer_idx]] + keys
            values = [frozen_values[layer_idx]] + values


Everything else in the update method (seq_ids handling, appending new tokens, etc.) can stay the same.

4. Parametrization implementations
4.1. Offset-based parametrization (current behavior)

Repackage existing logic into OffsetParametrization:

class OffsetParametrization(CartridgeParametrization):
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


This preserves current behavior, just moved into a class. Existing training should still work once the logging is updated to use cache.parametrization.logging_metrics().

4.2. MLP + residual parametrization

Now we can implement the MLP reparameterization from the paper snippet:

𝑃
𝑘
′
=
MLP
(
𝑃
𝑘
)
+
𝑃
𝑘
P
k
′
	​

=MLP(P
k
	​

)+P
k
	​


Interpretation: for each layer and each prefix token and head, we apply an MLP to the key/value vector and add a residual connection.

A flexible implementation:

Option A: share one MLP across all layers (like a global reparam)

Option B: per-layer MLPs.

I’ll outline shared-MLP for simplicity:

class ResidualMLPParametrization(CartridgeParametrization):
    class Config(ObjectConfig):
        _pass_as_config = True
        hidden_multiplier: float = 4.0  # width of the MLP
        activation: Literal["relu", "gelu"] = "relu"
        share_across_layers: bool = True

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

        def make_mlp():
            act = nn.ReLU() if self.cfg.activation == "relu" else nn.GELU()
            return nn.Sequential(
                nn.Linear(head_dim, hidden_dim),
                act,
                nn.Linear(hidden_dim, head_dim),
            )

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


That’s the core new parametrization; no changes anywhere else needed to use it.

5. Config / factory wiring

We want to be able to choose parametrization from config without touching training code.

5.1. Allow parametrization selection in the KV cache initializer

Extend KVCacheFactory.Config to carry “parametrization type + params”:

class KVCacheFactory(abc.ABC):
    class Config(ObjectConfig):
        _pass_as_config = True

        num_frozen_tokens: int = 1

        # New:
        parametrization_type: Literal["offset", "mlp_residual"] = "offset"
        parametrization_config: dict = Field(default_factory=dict)


In your concrete KV cache initializer’s initialize_kv_cache, once you have init_keys and init_values, you can do:

from cartridges.cache import OffsetParametrization, ResidualMLPParametrization

PARAM_REGISTRY = {
    "offset": OffsetParametrization,
    "mlp_residual": ResidualMLPParametrization,
}

def initialize_kv_cache(self, tokenizer, model, attn_config) -> TrainableCache:
    init_keys, init_values = ...  # your current code
    ParamCls = PARAM_REGISTRY[self.config.parametrization_type]

    if ParamCls is ResidualMLPParametrization:
        param_cfg = ResidualMLPParametrization.Config(**self.config.parametrization_config)
        parametrization = ParamCls(
            config=param_cfg,
            attn_config=attn_config,
            init_keys=init_keys,
            init_values=init_values,
            num_frozen_tokens=self.config.num_frozen_tokens,
        )
    else:
        parametrization = ParamCls(
            attn_config=attn_config,
            init_keys=init_keys,
            init_values=init_values,
            num_frozen_tokens=self.config.num_frozen_tokens,
        )

    return TrainableCache(
        config=attn_config,
        init_keys=init_keys,
        init_values=init_values,
        num_frozen_tokens=self.config.num_frozen_tokens,
        parametrization=parametrization,
    )


From the rest of the code’s perspective, it’s still just getting a TrainableCache.

6. Saving / loading & “compile-to-plain-prefix”

Right now TrainableCache.save writes materialized trainable_keys / trainable_values and frozen_keys / frozen_values. That works for deployment but loses parametrization structure and is not enough to resume training for MLPs.

You can support both use cases:

6.1. For deployment: keep the current behavior

For generation-only usage (Toka, etc.), saving materialized K/V is fine and even beneficial: after training, you can “bake in” the MLP into the cartridge and discard extra parameters (just as the paper suggests).

Implement a method on TrainableCache:

    def materialized_kv(self):
        # Returns the final K/V that would be prepended at runtime
        frozen_k, frozen_v = ([], [])
        train_k, train_v = ([], [])
        if self.parametrization is not None:
            frozen_k, frozen_v = self.parametrization.get_frozen()
            train_k, train_v = self.parametrization.get_trainable()

        mat_keys, mat_values = [], []
        for fk, tk in itertools.zip_longest(frozen_k, train_k, fillvalue=None):
            if fk is None and tk is None:
                mat_keys.append(None)
            elif fk is None:
                mat_keys.append(tk)
            elif tk is None:
                mat_keys.append(fk)
            else:
                mat_keys.append(torch.cat([fk, tk], dim=2))

        for fv, tv in itertools.zip_longest(frozen_v, train_v, fillvalue=None):
            if fv is None and tv is None:
                mat_values.append(None)
            elif fv is None:
                mat_values.append(tv)
            elif tv is None:
                mat_values.append(fv)
            else:
                mat_values.append(torch.cat([fv, tv], dim=2))

        return mat_keys, mat_values


Then save can stay mostly as-is:

    def save(self, path: str):
        trainable_keys, trainable_values = self.parametrization.get_trainable() if self.parametrization else ([], [])
        frozen_keys, frozen_values = self.parametrization.get_frozen() if self.parametrization else ([], [])

        torch.save(
            {
                "trainable_keys": trainable_keys,
                "trainable_values": trainable_values,
                "frozen_keys": frozen_keys,
                "frozen_values": frozen_values,
            },
            path,
        )


This is exactly the “after training, discard parametrization and keep projected prefix” behavior.

6.2. For resuming training: save full state (optional but nice)

If you want to resume training with the MLP (or any parametrization), also add:

    def save_full(self, path: str):
        torch.save(
            {
                "version": 2,
                "attn_config": self.config.__dict__,
                "num_frozen_tokens": self._num_frozen_tokens,
                "parametrization_type": self.parametrization.__class__.__name__,
                "state_dict": self.state_dict(),
            },
            path,
        )


and a corresponding from_full_pretrained. You can keep this separate from the existing from_pretrained used in KVCacheFactoryWithStateSaving to avoid breaking older checkpoints.

7. Minimal changes needed outside cache.py

Most code already treats the cache as a black box. The only tight couplings are:

Optimizer construction in train.py

optimizer = optim.Adam(
    wrapped_model.parameters() if use_peft else cache.parameters(), 
    lr=config.lr,
    weight_decay=config.weight_decay,
)


This continues to work: the parametrization is inside cache, so its parameters are included.

W&B logging in train.py

Currently:

key_norms = [cache.trainable_key_offsets[layer_idx].norm() for layer_idx in range(cache.config.n_layers)]
value_norms = [cache.trainable_value_offsets[layer_idx].norm() for layer_idx in range(cache.config.n_layers)]
reference_key_norms = [cache.reference_keys[layer_idx].norm() for layer_idx in range(cache.config.n_layers)]
reference_value_norms = [cache.reference_values[layer_idx].norm() for layer_idx in range(cache.config.n_layers)]


Replace this with the generic logging hook:

if hasattr(cache, "parametrization") and cache.parametrization is not None:
    metrics = cache.parametrization.logging_metrics()
else:
    metrics = {}

wandb.log(
    {
        "train/loss": accum_loss,
        "train/perplexity": torch.exp(accum_loss).item(),
        ...
        **{f"train/{k}": v for k, v in metrics.items()},
        **{f"optimizer/lr_group{i}": pg["lr"] for i, pg in enumerate(optimizer.param_groups)},
    },
    step=optimizer_step,
)


Now any parametrization can decide what to log, and training code stays generic.

Places where cache._num_trainable_tokens / cache._num_frozen_tokens are read

In training (for W&B logging and sanity checks).

In save_cache_to_toka_format we only care about kv_cache_initializer.max_tokens – no change.

In evaluation, we log num_cache_tokens via cache._num_trainable_tokens.

You can leave the _num_* attributes on TrainableCache as they are set in the constructor; they remain valid regardless of parametrization.

flex_generate and other generation paths

They only use TrainableCache.update, seq_ids, clear, etc. Those remain intact; parametrization is fully internal.

Summary

Conceptually, the plan is:

Step 1: Introduce CartridgeParametrization as a small, pure abstraction for “how K/V prefixes are parametrized”.

Step 2: Refactor TrainableCache to delegate all prefix logic to a CartridgeParametrization instance, while keeping its runtime KV-cache API unchanged.

Step 3: Move current “reference + offsets” behavior into OffsetParametrization (default) and add a ResidualMLPParametrization implementing 
𝑃
𝑘
′
=
MLP
(
𝑃
𝑘
)
+
𝑃
𝑘
P
k
′
	​

=MLP(P
k
	​

)+P
k
	​

.

Step 4: Add a parametrization selection hook to KVCacheFactory.Config and construct the appropriate parametrization in your initializer.

Step 5: Make logging generic via parametrization.logging_metrics() and keep current save semantics for deployment (materialized K/V). Optionally add save_full / from_full_pretrained for resuming training.

With this, you can add further parametrizations (low-rank, shared across layers, task-conditioned, etc.) by dropping in new CartridgeParametrization subclasses and wiring them through config—no changes to training / evaluation / generation code paths beyond the generic logging hook.