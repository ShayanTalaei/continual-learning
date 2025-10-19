python src/memory/distillation/convert_to_cartridges.py --input stalaei/distillation-dataset-test --output ./data/processed_dataset.parquet --model meta-llama/Llama-3.1-8B-Instruct --input-type huggingface --split train

python -m src.memory.distillation.distill_into_cartridge

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge


python -m src.memory.distillation.distill_into_cartridge .no_evals .init_from_text

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge .init_from_text

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge kv_cache.num_tokens=1 kv_cache.num_frozen_tokens=0 run_name=distill_1token


torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge .init_from_text run_name=init_from_history


torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge kv_cache.num_tokens=0 kv_cache.num_frozen_tokens=0 run_name=distill_0tokens


python -m src.memory.distillation.distill_into_cartridge kv_cache.num_tokens=0 kv_cache.num_frozen_tokens=0 run_name=distill_0tokens training.epochs=0 num_generate_problems=32


# Fixed runs

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge kv_cache.num_tokens=0 kv_cache.num_frozen_tokens=0 run_name=baseline_no_cartridge training.epochs=0


torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge .init_from_text run_name=init_from_icl_examples

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge .init_from_text run_name=init_from_icl_examples_64tokens kv_cache.num_tokens=64 

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge run_name=init_from_random

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge .init_from_text run_name=init_from_icl_examples_256tokens kv_cache.num_tokens=256

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge .init_from_text run_name=init_from_icl_cartridge kv_cache.init_text_file=/mnt/home/bradleyb/continual-learning/third_party/cartridges/examples/arxiv/cartridges.tex

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge .init_from_text run_name=init_from_extraicl_examples input_dataset.local_path=/mnt/data/shayan_memory/finer_train_data_gen_combined.jsonl

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge .init_from_text run_name=init_from_extraicl_examples_evaltemp0dot1 input_dataset.local_path=/mnt/data/shayan_memory/finer_train_data_gen_combined.jsonl generate_temperature=0.1

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge .init_from_text run_name=init_from_extraicl_examples_nosys input_dataset.local_path=/mnt/data/shayan_memory/finer_train_data_gen_combined.jsonl kv_cache.init_text_file=/mnt/home/bradleyb/continual-learning/src/memory/distillation/kv_cache_init_texts/v1_nosystem.txt


torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge .init_from_text run_name=init_from_extraicl_examples_256tokens input_dataset.local_path=/mnt/data/shayan_memory/finer_train_data_gen_combined.jsonl run_name=init_from_icl_examples_256tokens kv_cache.num_tokens=256

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge .init_from_text run_name=init_from_extraicl_examples_4096tokens input_dataset.local_path=/mnt/data/shayan_memory/finer_train_data_gen_combined.jsonl kv_cache.num_tokens=4096

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge .init_from_text run_name=init_from_extraicl_examples_nosys_4096tokens input_dataset.local_path=/mnt/data/shayan_memory/finer_train_data_gen_combined.jsonl kv_cache.num_tokens=4096 kv_cache.init_text_file=/mnt/home/bradleyb/continual-learning/src/memory/distillation/kv_cache_init_texts/v1_nosystem.txt

# Fixed runs

## Ablating the KV cache size

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge .init_from_text run_name=oct17_2048tokens_mem input_dataset.local_path=/mnt/data/shayan_memory/finer_train_data_gen_combined.jsonl kv_cache.num_tokens=2048 kv_cache.init_text_file=/mnt/home/bradleyb/continual-learning/src/memory/distillation/kv_cache_init_texts/v1_nosystem.txt training.train_temperature=1

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge .init_from_text run_name=oct17_256tokens_mem input_dataset.local_path=/mnt/data/shayan_memory/finer_train_data_gen_combined.jsonl kv_cache.num_tokens=256 kv_cache.init_text_file=/mnt/home/bradleyb/continual-learning/src/memory/distillation/kv_cache_init_texts/v1_nosystem.txt training.train_temperature=1 generate_batch_size=16

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge .init_from_text run_name=oct17_4096tokens_mem input_dataset.local_path=/mnt/data/shayan_memory/finer_train_data_gen_combined.jsonl kv_cache.num_tokens=4096 kv_cache.init_text_file=/mnt/home/bradleyb/continual-learning/src/memory/distillation/kv_cache_init_texts/v1_nosystem.txt training.train_temperature=1

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge .init_from_text run_name=oct17_8192tokens_mem input_dataset.local_path=/mnt/data/shayan_memory/finer_train_data_gen_combined.jsonl kv_cache.num_tokens=8192 kv_cache.init_text_file=/mnt/home/bradleyb/continual-learning/src/memory/distillation/kv_cache_init_texts/v1_nosystem.txt training.train_temperature=1 dataset.batch_size=4 dataset.packed_seq_length=16000 training.global_batch_size=16 generate_batch_size=8

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge .init_from_text run_name=oct17_16ktokens_mem input_dataset.local_path=/mnt/data/shayan_memory/finer_train_data_gen_combined.jsonl kv_cache.num_tokens=16384 kv_cache.init_text_file=/mnt/home/bradleyb/continual-learning/src/memory/distillation/kv_cache_init_texts/v1_nosystem.txt training.train_temperature=1 dataset.batch_size=4 dataset.packed_seq_length=16000 training.global_batch_size=16 generate_batch_size=8


## Ablating the KV cache init

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge .init_from_text run_name=oct17_2048tokens_sysmem input_dataset.local_path=/mnt/data/shayan_memory/finer_train_data_gen_combined.jsonl kv_cache.num_tokens=2048 kv_cache.init_text_file=/mnt/home/bradleyb/continual-learning/src/memory/distillation/kv_cache_init_texts/v1.txt training.train_temperature=1

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge .init_from_text run_name=oct17_2048tokens_cartridge input_dataset.local_path=/mnt/data/shayan_memory/finer_train_data_gen_combined.jsonl kv_cache.num_tokens=2048 training.train_temperature=1 kv_cache.init_text_file=/mnt/home/bradleyb/continual-learning/third_party/cartridges/examples/arxiv/cartridges.tex


python -m src.memory.distillation.distill_into_cartridge .no_evals .init_from_text input_dataset.local_path=/mnt/data/shayan_memory/finer_data_gen_shuffled_16x100/dataset.jsonl kv_cache.num_tokens=2048 training.train_temperature=1 kv_cache.init_text_file=/mnt/home/bradleyb/continual-learning/third_party/cartridges/examples/arxiv/cartridges.tex input_dataset.filter_incorrect=T


# Running and filtering out incorrect

python -m src.memory.distillation.distill_into_cartridge .init_from_text run_name=oct18_2048tokens_sysmem_filterincorrect input_dataset.local_path=/mnt/data/shayan_memory/finer_train_data_gen_combined_with_evaluation.jsonl kv_cache.num_tokens=2048 kv_cache.init_text_file=/mnt/home/bradleyb/continual-learning/src/memory/distillation/kv_cache_init_texts/v1.txt training.train_temperature=1 input_dataset.filter_incorrect=T
torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge .init_from_text run_name=oct18_2048tokens_sysmem_filterincorrect input_dataset.local_path=/mnt/data/shayan_memory/finer_train_data_gen_combined_with_evaluation.jsonl kv_cache.num_tokens=2048 kv_cache.init_text_file=/mnt/home/bradleyb/continual-learning/src/memory/distillation/kv_cache_init_texts/v1.txt training.train_temperature=1 input_dataset.filter_incorrect=T

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge .init_from_text run_name=oct17_2048tokens_memwithsystags input_dataset.local_path=/mnt/data/shayan_memory/finer_train_data_gen_combined.jsonl kv_cache.num_tokens=2048 kv_cache.init_text_file=/mnt/home/bradleyb/continual-learning/src/memory/distillation/kv_cache_init_texts/v1_mem_with_syschat_tags.txt training.train_temperature=1

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge .init_from_text run_name=oct18_2048tokens_sysmem_gttargs input_dataset.local_path=/mnt/data/shayan_memory/finer_train_data_gen_combined_with_evaluation.jsonl kv_cache.num_tokens=2048 kv_cache.init_text_file=/mnt/home/bradleyb/continual-learning/src/memory/distillation/kv_cache_init_texts/v1.txt training.train_temperature=1 input_dataset.ground_truth_target=T
