python src/memory/distillation/convert_to_cartridges.py --input stalaei/distillation-dataset-test --output ./data/processed_dataset.parquet --model meta-llama/Llama-3.1-8B-Instruct --input-type huggingface --split train

python -m src.memory.distillation.distill_into_cartridge

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge


python -m src.memory.distillation.distill_into_cartridge .no_evals .init_from_text

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge .init_from_text

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge kv_cache.num_tokens=16 kv_cache.num_frozen_tokens=0 run_name=distill_16tokens