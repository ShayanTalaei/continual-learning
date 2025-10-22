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
torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge .init_from_text run_name=oct18_2048tokens_sysmem_filterincorrect_5e4lr input_dataset.local_path=/mnt/data/shayan_memory/finer_train_data_gen_combined_with_evaluation.jsonl kv_cache.num_tokens=2048 kv_cache.init_text_file=/mnt/home/bradleyb/continual-learning/src/memory/distillation/kv_cache_init_texts/v1.txt training.train_temperature=1 input_dataset.filter_incorrect=T training.lr=5e-4

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge .init_from_text run_name=oct18_2048tokens_sysmem_filterincorrect input_dataset.local_path=/mnt/data/shayan_memory/finer_train_data_gen_combined_with_evaluation.jsonl kv_cache.num_tokens=2048 kv_cache.init_text_file=/mnt/home/bradleyb/continual-learning/src/memory/distillation/kv_cache_init_texts/v1.txt training.train_temperature=1 input_dataset.filter_incorrect=T

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge .init_from_text run_name=oct17_2048tokens_memwithsystags input_dataset.local_path=/mnt/data/shayan_memory/finer_train_data_gen_combined.jsonl kv_cache.num_tokens=2048 kv_cache.init_text_file=/mnt/home/bradleyb/continual-learning/src/memory/distillation/kv_cache_init_texts/v1_mem_with_syschat_tags.txt training.train_temperature=1

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge .init_from_text run_name=oct18_2048tokens_sysmem_gttargs input_dataset.local_path=/mnt/data/shayan_memory/finer_train_data_gen_combined_with_evaluation.jsonl kv_cache.num_tokens=2048 kv_cache.init_text_file=/mnt/home/bradleyb/continual-learning/src/memory/distillation/kv_cache_init_texts/v1.txt training.train_temperature=1 input_dataset.ground_truth_target=T

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge .init_from_text run_name=oct18_128tokens_sysmem_gttargs input_dataset.local_path=/mnt/data/shayan_memory/finer_train_data_gen_combined_with_evaluation.jsonl kv_cache.num_tokens=128 kv_cache.init_text_file=/mnt/home/bradleyb/continual-learning/src/memory/distillation/kv_cache_init_texts/v1.txt training.train_temperature=1 input_dataset.ground_truth_target=T


torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge .init_from_text run_name=oct18_16x100_2048tokens_sysmem_gttargs input_dataset.local_path=/mnt/data/shayan_memory/finer_data_gen_shuffled_16x100/dataset.jsonl kv_cache.num_tokens=2048 kv_cache.init_text_file=/mnt/home/bradleyb/continual-learning/src/memory/distillation/kv_cache_init_texts/v1.txt training.train_temperature=1 input_dataset.ground_truth_target=T


## Hyperparameter ablations


## LRs

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge .init_from_text run_name=oct18_2048tokens_sysmem_filterincorrect_1e3lr input_dataset.local_path=/mnt/data/shayan_memory/finer_train_data_gen_combined_with_evaluation.jsonl kv_cache.num_tokens=2048 kv_cache.init_text_file=/mnt/home/bradleyb/continual-learning/src/memory/distillation/kv_cache_init_texts/v1.txt training.train_temperature=1 input_dataset.filter_incorrect=T training.lr=1e-3

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge .init_from_text run_name=oct18_2048tokens_sysmem_filterincorrect_1e3lr input_dataset.local_path=/mnt/data/shayan_memory/finer_train_data_gen_combined_with_evaluation.jsonl kv_cache.num_tokens=2048 kv_cache.init_text_file=/mnt/home/bradleyb/continual-learning/src/memory/distillation/kv_cache_init_texts/v1.txt training.train_temperature=1 input_dataset.filter_incorrect=T training.lr=5e-5


## Weight decay

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge .init_from_text run_name=oct18_2048tokens_sysmem_filterincorrect_01wd input_dataset.local_path=/mnt/data/shayan_memory/finer_train_data_gen_combined_with_evaluation.jsonl kv_cache.num_tokens=2048 kv_cache.init_text_file=/mnt/home/bradleyb/continual-learning/src/memory/distillation/kv_cache_init_texts/v1.txt training.train_temperature=1 input_dataset.filter_incorrect=T training.weight_decay=0.1


## KV cache size

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge .init_from_text run_name=oct18_128tokens_sysmem_filterincorrect input_dataset.local_path=/mnt/data/shayan_memory/finer_train_data_gen_combined_with_evaluation.jsonl kv_cache.num_tokens=128 kv_cache.init_text_file=/mnt/home/bradleyb/continual-learning/src/memory/distillation/kv_cache_init_texts/v1.txt training.train_temperature=1 input_dataset.filter_incorrect=T

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge .init_from_text run_name=oct18_256tokens_sysmem_filterincorrect input_dataset.local_path=/mnt/data/shayan_memory/finer_train_data_gen_combined_with_evaluation.jsonl kv_cache.num_tokens=256 kv_cache.init_text_file=/mnt/home/bradleyb/continual-learning/src/memory/distillation/kv_cache_init_texts/v1.txt training.train_temperature=1 input_dataset.filter_incorrect=T


# Back to Marlowe

## Checking environment


torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=oct21_100train_128tokens_sysmem_filterincorrect \
    kv_cache.num_tokens=128 \
    training.train_temperature=1 \
    input_dataset.filter_incorrect=T \
    .init_from_text \
    kv_cache.init_text_file=src/memory/distillation/kv_cache_init_texts/v1.txt \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/finer_v1_train_ICL_exclude_current_subsampling_50_temp_0.7/dataset_filtered.jsonl \
    val_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/finer_v1_val_full_memory_1000/dataset.jsonl \
    system_prompt_path=/scratch/m000122/bcabrown/continual-learning/src/data/prompts/finer/system_prompt_brad_magic.txt

## Test streaming

torchrun --nproc_per_node 8 

## n31
python -m src.memory.distillation.distill_into_cartridge \
    run_name=oct21_6400train_128tokens_sysmem_filterincorrect \
    kv_cache.num_tokens=128 \
    training.train_temperature=1 \
    input_dataset.filter_incorrect=T \
    .init_from_text \
    kv_cache.init_text_file=src/memory/distillation/kv_cache_init_texts/v1.txt \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/finer_v1_large_train_random_50_subsampling_16x6400_temp_0.7/dataset_filtered.jsonl \
    val_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/finer_v1_val_full_memory_1000/dataset.jsonl \
    system_prompt_path=src/data/prompts/finer/system_prompt_brad_magic.txt \
    training.lr=5e-4 \
    generate_eval_every_n_steps=50

## n06
torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=oct20_6400train_128tokens_sysmem_filterincorrect \
    kv_cache.num_tokens=128 \
    training.train_temperature=1 \
    input_dataset.filter_incorrect=T \
    .init_from_text \
    kv_cache.init_text_file=src/memory/distillation/kv_cache_init_texts/v1.txt \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/finer_v1_large_train_random_50_subsampling_16x6400_temp_0.7/dataset_filtered.jsonl \
    val_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/finer_v1_val_full_memory_1000/dataset.jsonl \
    system_prompt_path=src/data/prompts/finer/system_prompt_brad_magic.txt \
    training.lr=5e-4 \
    generate_eval_every_n_steps=50


# Overnight runs on October 20

## n31
torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=oct21_100train_128tokens_sysmem_filterincorrect \
    kv_cache.num_tokens=128 \
    training.train_temperature=1 \
    input_dataset.filter_incorrect=T \
    .init_from_text \
    kv_cache.init_text_file=src/memory/distillation/kv_cache_init_texts/v1.txt \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/finer_v1_train_ICL_exclude_current_subsampling_50_temp_0.7/dataset_filtered.jsonl \
    val_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/finer_v1_val_full_memory_1000/dataset.jsonl \
    system_prompt_path=src/data/prompts/finer/system_prompt_brad_magic.txt \
    training.lr=5e-4 \
    generate_eval_every_n_steps=50

## n06
torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=oct20_6400train_128tokens_sysmem_filterincorrect \
    kv_cache.num_tokens=128 \
    training.train_temperature=1 \
    input_dataset.filter_incorrect=T \
    .init_from_text \
    kv_cache.init_text_file=src/memory/distillation/kv_cache_init_texts/v1.txt \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/finer_v1_large_train_random_50_subsampling_16x6400_temp_0.7/dataset_filtered.jsonl \
    val_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/finer_v1_val_full_memory_1000/dataset.jsonl \
    system_prompt_path=src/data/prompts/finer/system_prompt_brad_magic.txt \
    training.lr=5e-4 \
    generate_eval_every_n_steps=50

## matx2
TRANSFORMERS_NO_FLASH_ATTN=1 torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge \
    run_name=oct21_100train_2048tokens_sysmem_filterincorrect \
    kv_cache.num_tokens=2048 \
    training.train_temperature=1 \
    input_dataset.filter_incorrect=T \
    .init_from_text \
    kv_cache.init_text_file=src/memory/distillation/kv_cache_init_texts/v1.txt \
    input_dataset.local_path=/matx/u/bcabrown/shayan_memory/data/finer_v1_train_ICL_exclude_current_subsampling_50_temp_0.7_filtered.jsonl \
    val_dataset.local_path=/matx/u/bcabrown/shayan_memory/data/finer_v1_val_full_memory_1000.jsonl \
    system_prompt_path=src/data/prompts/finer/system_prompt_brad_magic.txt \
    .matx \
    generate_eval_every_n_steps=50

## matx3
TRANSFORMERS_NO_FLASH_ATTN=1 torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge \
    run_name=oct21_6400train_2048tokens_sysmem_filterincorrect \
    kv_cache.num_tokens=2048 \
    training.train_temperature=1 \
    input_dataset.filter_incorrect=T \
    .init_from_text \
    kv_cache.init_text_file=src/memory/distillation/kv_cache_init_texts/v1.txt \
    input_dataset.local_path=/matx/u/bcabrown/shayan_memory/data/finer_v1_large_train_random_50_subsampling_16x6400_temp_0.7.jsonl \
    val_dataset.local_path=/matx/u/bcabrown/shayan_memory/data/finer_v1_val_full_memory_1000.jsonl \
    system_prompt_path=src/data/prompts/finer/system_prompt_brad_magic.txt \
    .matx \
    generate_eval_every_n_steps=50


# Training without logit distillation, just sft

## n07
torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge \
    run_name=oct21_100train_128tokens_sysmem_filterincorrect_tokensupervision \
    kv_cache.num_tokens=128 \
    training.train_temperature=1 \
    input_dataset.filter_incorrect=T \
    .init_from_text \
    kv_cache.init_text_file=src/memory/distillation/kv_cache_init_texts/v1.txt \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/finer_v1_train_ICL_exclude_current_subsampling_50_temp_0.7/dataset_filtered.jsonl \
    val_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/finer_v1_val_full_memory_1000/dataset.jsonl \
    system_prompt_path=src/data/prompts/finer/system_prompt_brad_magic.txt \
    training.lr=5e-4 \
    generate_eval_every_n_steps=50 \
    training.train_without_logits=T

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge \
    run_name=oct21_100train_128tokens_sysmem \
    kv_cache.num_tokens=128 \
    training.train_temperature=1 \
    input_dataset.filter_incorrect=T \
    .init_from_text \
    kv_cache.init_text_file=src/memory/distillation/kv_cache_init_texts/v1.txt \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/finer_v1_train_ICL_exclude_current_subsampling_50_temp_0.7/dataset.jsonl \
    val_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/finer_v1_val_full_memory_1000/dataset.jsonl \
    system_prompt_path=src/data/prompts/finer/system_prompt_brad_magic.txt \
    training.lr=5e-4 \
    generate_eval_every_n_steps=50 \
    training.train_without_logits=F

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge \
    run_name=oct21_100train_2048tokens_sysmem_tokensupervision \
    kv_cache.num_tokens=2048 \
    training.train_temperature=1 \
    input_dataset.filter_incorrect=T \
    .init_from_text \
    kv_cache.init_text_file=src/memory/distillation/kv_cache_init_texts/v1.txt \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/finer_v1_train_ICL_exclude_current_subsampling_50_temp_0.7/dataset.jsonl \
    val_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/finer_v1_val_full_memory_1000/dataset.jsonl \
    system_prompt_path=src/data/prompts/finer/system_prompt_brad_magic.txt \
    training.lr=5e-4 \
    generate_eval_every_n_steps=50 \
    training.train_without_logits=F

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge \
    run_name=oct21_100train_2048tokens_sysmem \
    kv_cache.num_tokens=2048 \
    training.train_temperature=1 \
    input_dataset.filter_incorrect=T \
    .init_from_text \
    kv_cache.init_text_file=src/memory/distillation/kv_cache_init_texts/v1.txt \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/finer_v1_train_ICL_exclude_current_subsampling_50_temp_0.7/dataset.jsonl \
    val_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/finer_v1_val_full_memory_1000/dataset.jsonl \
    system_prompt_path=src/data/prompts/finer/system_prompt_brad_magic.txt \
    training.lr=5e-4 \
    generate_eval_every_n_steps=50

torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge \
    run_name=oct21_100train_128tokens_sysmem_fixed \
    kv_cache.num_tokens=128 \
    training.train_temperature=1 \
    input_dataset.filter_incorrect=F \
    .init_from_text \
    kv_cache.init_text_file=src/memory/distillation/kv_cache_init_texts/v1.txt \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/finer_v1_train_ICL_exclude_current_subsampling_50_temp_0.7/dataset.jsonl \
    val_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/finer_v1_val_full_memory_1000/dataset.jsonl \
    system_prompt_path=src/data/prompts/finer/system_prompt_brad_magic.txt \
    training.lr=5e-4 \
    generate_eval_every_n_steps=50


# matx3
torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge \
    run_name=oct21_6400train_128tokens_sysmem_filterincorrect_tokensupervision \
    kv_cache.num_tokens=128 \
    training.train_temperature=1 \
    input_dataset.filter_incorrect=T \
    .init_from_text \
    kv_cache.init_text_file=src/memory/distillation/kv_cache_init_texts/v1.txt \
    input_dataset.local_path=/matx/u/bcabrown/shayan_memory/data/finer_v1_large_train_random_50_subsampling_16x6400_temp_0.7.jsonl \
    val_dataset.local_path=/matx/u/bcabrown/shayan_memory/data/finer_v1_val_full_memory_1000.jsonl \
    system_prompt_path=src/data/prompts/finer/system_prompt_brad_magic.txt \
    .matx \
    training.lr=5e-4 \
    generate_eval_every_n_steps=50 \
    training.train_without_logits=T


torchrun --nproc_per_node 8 -m src.memory.distillation.distill_into_cartridge \
    run_name=oct21_100train_128tokens_sysmem_filterincorrect_tokensupervision \
    kv_cache.num_tokens=128 \
    training.train_temperature=1 \
    input_dataset.filter_incorrect=T \
    .init_from_text \
    kv_cache.init_text_file=src/memory/distillation/kv_cache_init_texts/v1.txt \
    input_dataset.local_path=/matx/u/bcabrown/shayan_memory/data/finer_v1_train_ICL_exclude_current_subsampling_50_temp_0.7_filtered.jsonl \
    val_dataset.local_path=/matx/u/bcabrown/shayan_memory/data/finer_v1_val_full_memory_1000.jsonl \
    system_prompt_path=src/data/prompts/finer/system_prompt_brad_magic.txt \
    .matx \
    training.lr=5e-4 \
    generate_eval_every_n_steps=50 \
    training.train_without_logits=T

    