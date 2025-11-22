import os
from pathlib import Path
import pydrantic

from cartridges.initialization import KVFromText
from cartridges.train import TrainConfig
from cartridges.models import HFModelConfig, FlexLlamaForCausalLM
from cartridges.datasets import TrainDataset, DataSource

# Set environment variables
os.environ.setdefault("CARTRIDGES_DIR", str(Path(__file__).parent.parent.parent))
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", str(Path(__file__).parent.parent.parent / "outputs"))

config = TrainConfig(
    name="amd-10k-cartridge",
    
    # Model configuration (Llama 3)
    model=HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        model_cls=FlexLlamaForCausalLM,
        tuning_method="custom_prefix",
    ),
    
    # Initialize KV cache from AMD 10-K text (FinanceBench)
    kv_cache_initializer=KVFromText.Config(
        text_source=os.path.join(os.environ["CARTRIDGES_DIR"], "data/10k/amd_10k_financebench.txt"),
        max_tokens=2048,
    ),
    
    # Training dataset (synthesized Q&A pairs)
    dataset=TrainDataset.Config(
        data_sources=[
            DataSource(
                # NOTE: this file is produced by amd_synthesize.py
                # You should set AMD_SYNTH_DATASET_PATH to the 'dataset.parquet'
                # path printed at the end of the synthesis run.
                path=os.environ.get(
                    "AMD_SYNTH_DATASET_PATH",
                    os.path.join(
                        os.environ["CARTRIDGES_DIR"],
                        "path",
                        "to",
                        "amd_dataset.parquet",
                    ),
                ),
                type="local",
            ),
        ],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ),
    
    # Training hyperparameters
    lr=2e-2,
    epochs=3,
    global_batch_size=16,  # Smaller batch size for single GPU
    
    save_every_n_steps=128,
    save_after_training=True,
    distributed_backend="gloo",  # Use gloo for single GPU
    
    # No wandb logging
    wandb=None,
)

if __name__ == "__main__":
    pydrantic.main(config)
