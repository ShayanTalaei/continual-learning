import os
from pathlib import Path
import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.data.chunkers import TokenChunker
from cartridges.data.resources import TextFileResource
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.clients.tokasaurus import TokasaurusClient

# Set environment variable for cartridges directory
os.environ.setdefault("CARTRIDGES_DIR", str(Path(__file__).parent.parent.parent))

# Use Llama 3 for synthesis
client = TokasaurusClient.Config(
    url=os.environ.get("TOKASAURUS_URL", "http://localhost:10210"),
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

config = SynthesizeConfig(
    synthesizer=SelfStudySynthesizer.Config(
        client=client,
        max_rounds=1,
        prob_thinking=0.75,
        tools=[],
        resources=[
            TextFileResource.Config(
                path=os.path.join(os.environ["CARTRIDGES_DIR"], "data/10k/pepsi_10k_financebench.txt"),
                seed_prompts=[
                    "question",        # Generate questions about the content
                    "summarization",   # Summarize sections
                    "structuring",     # Extract structured information
                    # "aggregation",     # Aggregate / analyze information
                    "use_case",        # Compare / apply information
                ],
                chunker=TokenChunker.Config(
                    tokenizer=client.model_name,
                    min_tokens_per_chunk=None,   # Fixed size chunks
                    max_tokens_per_chunk=8192,   # 8192 tokens per chunk
                ),
            )
        ],
    ),
    wandb=None,
    num_samples=20000,  # Generate 512 Q&A pairs
    batch_size=1,
    max_num_batches_in_parallel=128,

    name=FormatStringVariable("pepsi_10k_synthesize_{synthesizer.client.model_name}_n{num_samples}"),
    run_id=FormatStringVariable("{name}"),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", str(Path(__file__).parent.parent.parent / "outputs")),
    
    upload_to_wandb=False,
    save_wandb_preview=False,
    upload_to_hf=False,
)

if __name__ == "__main__":
    pydrantic.main([config])
