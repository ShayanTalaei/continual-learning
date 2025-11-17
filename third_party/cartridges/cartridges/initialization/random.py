
import torch
from cartridges.cache import AttnConfig, KVCacheFactory, TrainableCache, create_parametrization

class KVFromRandomVectors(KVCacheFactory):
    class Config(KVCacheFactory.Config):
        max_tokens: int

    def __init__(self, config: Config):
        self.config = config

    def initialize_kv_cache(
        self,
        tokenizer,
        model,
        attn_config: AttnConfig,
    ) -> TrainableCache:
        rand_vectors = lambda: [
            torch.randn(
                1, attn_config.n_heads, self.config.max_tokens, attn_config.head_dim,
                dtype=torch.bfloat16,
            )
            for _ in range(attn_config.n_layers)
        ]

        init_keys = rand_vectors()
        init_values = rand_vectors()

        parametrization = create_parametrization(
            parametrization_type=self.config.parametrization_type,
            parametrization_config=self.config.parametrization_config,
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
