import os
from pathlib import Path
import pydrantic

from cartridges.initialization import KVFromText
from cartridges.train import TrainConfig, GenerationEvalConfig
from cartridges.models import HFModelConfig, FlexLlamaForCausalLM
from cartridges.datasets import TrainDataset, GenerateEvalDataset, DataSource
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
    name="pepsi-10k-cartridge-gemini-eval",

    # Model configuration (Llama 3)
    model=HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        model_cls=FlexLlamaForCausalLM,
        tuning_method="custom_prefix",
    ),

    # Initialize KV cache from PepsiCo 10-K text (FinanceBench)
    kv_cache_initializer=KVFromText.Config(
        text_source=os.path.join(
            os.environ["CARTRIDGES_DIR"],
            "data/10k/pepsi_10k_financebench.txt",
        ),
        max_tokens=2048,
    ),

    # Training dataset (synthesized Q&A pairs)
    dataset=TrainDataset.Config(
        data_sources=[
            DataSource(
                # produced by pepsi_synthesize.py
                path=os.environ.get(
                    "PEPSI_SYNTH_DATASET_PATH",
                    os.path.join(
                        os.environ["CARTRIDGES_DIR"],
                        "path",
                        "to",
                        "pepsi_dataset.parquet",
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
    global_batch_size=32,

    save_every_n_steps=128,
    save_after_training=True,
    distributed_backend="gloo",  # Use gloo for single GPU

    # Generation eval on Gemini QA conversations
    generate_eval_every_n_steps=128,
    generate_evals=[
        GenerationEvalConfig(
            dataset=GenerateEvalDataset.Config(
                data_source=DataSource(
                    path=str(DATA_DIR / "eval" / "pepsi_qa_gemini.parquet"),
                    type="local",
                ),
            ),
            name_for_wandb="pepsi_gemini_generation_eval",
            batch_size=16,
            generate_max_new_tokens=256,
            temperature=1.0,
        ),
    ],

    wandb=WandBConfig(tags=["train", "pepsi10k"]),
)


if __name__ == "__main__":
    pydrantic.main([config])
