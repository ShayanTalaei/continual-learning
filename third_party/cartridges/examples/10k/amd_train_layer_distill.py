import os
from pathlib import Path
import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.initialization import KVFromText
from cartridges.train import TrainConfig, LossEvalConfig, GenerationEvalConfig, LayerDistillConfig
from cartridges.models import HFModelConfig, FlexLlamaForCausalLM
from cartridges.datasets import TrainDataset, LossEvalDataset, DataSource, GenerateEvalDataset
from cartridges.utils.wandb import WandBConfig

# Set environment variables
os.environ.setdefault("CARTRIDGES_DIR", str(Path(__file__).parent.parent.parent))
os.environ.setdefault(
    "CARTRIDGES_OUTPUT_DIR",
    str(Path(__file__).parent.parent.parent / "outputs"),
)

CARTRIDGES_DIR = Path(os.environ["CARTRIDGES_DIR"])
DATA_DIR = CARTRIDGES_DIR / "data" / "10k"


config = TrainConfig(
    name="amd-10k-cartridge-layer-distill",

    # Model configuration (Llama 3)
    model=HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        model_cls=FlexLlamaForCausalLM,
        # tuning_method="custom_prefix",
    ),

    # Initialize KV cache from AMD 10-K text (FinanceBench)
    kv_cache_initializer=KVFromText.Config(
        text_source=os.path.join(
            os.environ["CARTRIDGES_DIR"],
            "data/10k/amd_10k_financebench.txt",
        ),
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
                        "output",
                        "amd_synthesize",
                        "meta-llama",
                        "Llama-3.2-3B-Instruct_n65536-0",
                        "artifact",
                        "dataset.parquet",
                    ),
                ),
                type="local",
            ),
        ],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ),

    # Training hyperparam/ters
    lr=2e-2,
    epochs=1,
    global_batch_size=32,  # Smaller batch size for single GPU

    save_every_n_steps=128,
    save_after_training=True,
    distributed_backend="gloo",  # Use gloo for single GPU

    # === LAYER DISTILLATION CONFIG ===
    # Enable layer-wise distillation (teacher forward with real context)
    use_layer_distill=True,




    layer_distill=LayerDistillConfig(
        # Which signals to distill: "post_attn" (after attention), "post_mlp" (layer output)
        signals=["post_mlp"],
        # Which layers: "all" or list of layer indices like [0, 8, 16, 24]
        layers="all",
        # Loss type: "mse" or "cosine"
        loss_type="mse",
        # Weight for layer loss (relative to logit distillation loss)
        weight=1.0,
    ),

    # Loss evaluation on Gemini QA conversations
    generate_eval_every_n_steps=512,
    generate_evals=[
        GenerationEvalConfig(
            dataset=GenerateEvalDataset.Config(
                data_source=DataSource(
                    path=str(DATA_DIR / "eval" / "amd_qa_gemini.parquet"),
                    type="local",
                ),
            ),
            name_for_wandb="amd_gemini_loss_eval_31-28",
            batch_size=16
        ),
    ],
    wandb=WandBConfig(tags=["train", "amd10k", "layer-distill"]),
)


if __name__ == "__main__":
    pydrantic.main([config])
