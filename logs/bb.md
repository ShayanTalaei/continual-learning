python src/memory/distillation/convert_to_cartridges.py --input stalaei/distillation-dataset-test --output ./data/processed_dataset.parquet --model meta-llama/Llama-3.1-8B-Instruct --input-type huggingface --split train

python -m src.memory.distillation.distill_into_cartridge

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge --config configs/distillation/cartridge_train_example.yaml