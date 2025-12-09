## Launchig Toka

Installing these versions in the env:
>>> flashinfer.__version__
'0.2.2.post1'
>>> torch.__version__
'2.8.0+cu128'


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 toka model=meta-llama/Llama-3.1-8B-Instruct \
     dp_size=8 \
     port=8096 \
     torch_compile=T \
     kv_cache_num_tokens='(400000)' \
     max_tokens_per_forward='(128*1024)' \
     max_seqs_per_forward=128 \
     use_hydragen=True \
     hydragen_min_prefix_len=512 \
     hydragen_min_group_size=32 \
     cudagraph_max_size=16 \
     stats_report_seconds=1 \
     max_topk_logprobs=50 \
     cartridge_dir=/scratch/m000122/stalaei/continual-learning/cartridges


## Running history agent
- Make sure that the toka port is matching the server.

python -m src.main --config configs/finer/history_list_toka_llama8b.yaml


# Training cartridge on SynthCity task

## Delta node

## Chat format data with boxed only

torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov4_chatboxed \
    kv_cache.num_tokens=128 \
    training.train_temperature=1 \
    .init_from_text \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    .toka \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/projects/bfsg/stalaei/logs/continual_learning/data/cities_easy_synthetic_gen_l8b_non_collapsed_boxed_only_20000_with_subsample_and_original_experiences/dataset.jsonl \
    output.local_dir=/projects/bfsg/stalaei/continual-learning/cartridges/

### Dataset with new chat as distractors


torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov4_chatboxed_with_gt_distractors \
    kv_cache.num_tokens=128 \
    training.train_temperature=1 \
    .init_from_text \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    .toka \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/projects/bfsg/stalaei/logs/continual_learning/data/cities_easy_synthetic_gen_l8b_non_collapsed_boxed_only_20000_with_subsample_and_original_experiences/dataset_with_new_cities_distractors.jsonl \
    output.local_dir=/projects/bfsg/stalaei/continual-learning/cartridges/


### initializing from another cartridge

torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov4_chatboxed_with_gt_distractors_cartridge_init \
    kv_cache.num_tokens=128 \
    training.train_temperature=1 \
    load_cache_path=/projects/bfsg/stalaei/continual-learning/cartridges/nov4_chatboxed-cache-step600/cartridge.pt \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    .toka \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/projects/bfsg/stalaei/logs/continual_learning/data/cities_easy_synthetic_gen_l8b_non_collapsed_boxed_only_20000_with_subsample_and_original_experiences/dataset_with_new_cities_distractors.jsonl \
    output.local_dir=/projects/bfsg/stalaei/continual-learning/cartridges/

torchrun --nproc_per_node 2 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov5_chatboxed_with_up_to_5_gt_distractors_cartridge_init \
    kv_cache.num_tokens=128 \
    training.train_temperature=1 \
    load_cache_path=/projects/bfsg/stalaei/continual-learning/cartridges/nov4_chatboxed-cache-step600/cartridge.pt \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    .toka \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/projects/bfsg/stalaei/logs/continual_learning/data/cities_easy_synthetic_gen_l8b_non_collapsed_boxed_only_20000_with_subsample_and_original_experiences/dataset_with_up_to_5_new_cities_distractors.jsonl \
    output.local_dir=/projects/bfsg/stalaei/continual-learning/cartridges/


## gh089
torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov5_chatboxed_with_up_to_5_gt_distractors_cartridge_init \
    kv_cache.num_tokens=128 \
    training.train_temperature=1 \
    load_cache_path=/projects/bfsg/stalaei/continual-learning/cartridges/nov4_chatboxed-cache-step600/cartridge.pt \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    .toka \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/projects/bfsg/stalaei/logs/continual_learning/data/cities_easy_synthetic_gen_l8b_non_collapsed_boxed_only_20000_with_subsample_and_original_experiences/dataset_with_up_to_5_new_cities_distractors.jsonl \
    output.local_dir=/projects/bfsg/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=5 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/u/stalaei/code/continual-learning/notebooks/experiences.jsonl \
    gen_val_num_repeats=10

## gh136
torchrun --nproc_per_node 2 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov5_chatboxed_with_up_to_5_gt_distractors_text_init \
    kv_cache.num_tokens=128 \
    training.train_temperature=1 \
    .init_from_text \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    .toka \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/projects/bfsg/stalaei/logs/continual_learning/data/cities_easy_synthetic_gen_l8b_non_collapsed_boxed_only_20000_with_subsample_and_original_experiences/dataset_with_up_to_5_new_cities_distractors.jsonl \
    output.local_dir=/projects/bfsg/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=5 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/u/stalaei/code/continual-learning/notebooks/experiences.jsonl \
    gen_val_num_repeats=10


## gh089
torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov5_chatboxed_with_up_to_5_gt_distractors_text_init_fixed \
    kv_cache.num_tokens=128 \
    training.train_temperature=1 \
    .init_from_text \ 
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    .toka \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/projects/bfsg/stalaei/logs/continual_learning/data/cities_easy_synthetic_gen_l8b_non_collapsed_boxed_only_20000_with_subsample_and_original_experiences/dataset_with_up_to_5_new_cities_distractors.jsonl \
    output.local_dir=/projects/bfsg/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=5 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/u/stalaei/code/continual-learning/notebooks/experiences.jsonl \
    gen_val_num_repeats=10


torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov5_chatboxed_with_up_to_500_gt_distractors_text_init_fixed \
    kv_cache.num_tokens=128 \
    training.train_temperature=1 \
    .init_from_text \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    .toka \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/projects/bfsg/stalaei/logs/continual_learning/data/cities_easy_synthetic_gen_l8b_non_collapsed_boxed_only_20000_with_subsample_and_original_experiences/dataset_with_up_to_500_new_cities_distractors.jsonl \
    output.local_dir=/projects/bfsg/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=5 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/u/stalaei/code/continual-learning/notebooks/experiences.jsonl \
    gen_val_num_repeats=10

## gh039
torchrun --nproc_per_node 4 -m src.memory.distillation.di
still_into_cartridge \
    run_name=nov5_chatboxed_with_up_to_50_gt_distractors_text_init_fixed \ 
    kv_cache.num_tokens=128 \
    training.train_temperature=1 \
    .init_from_text \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    .toka \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/projects/bfsg/stalaei/logs/continual_learning/data/cities_easy_synthetic_gen_l8b_non_collapsed_box
ed_only_20000_with_subsample_and_original_experiences/dataset_with_up_to_50_new_cities_distractors.jsonl \ 
    output.local_dir=/projects/bfsg/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/u/stalaei/code/continual-learning/notebooks/experiences.jsonl \
    gen_val_num_repeats=10 .long_seqs

## Count the number of tokens
python scripts/compute_token_stats.py /projects/bfsg/stalaei/logs/continual_learning/data/cities_easy_synthetic_gen_l8b_non_collapsed_boxed_only_20000_with_subsample_and_original_experiences/dataset_with_up_to_50_new_cities_distractors.jsonl --model-name meta-llama/Llama-3.1-8B-Instruct --system-prompt src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt

## Nov 6

torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov6_chatboxed_with_up_to_50_gt_distractors_text_init_fixed kv_cache.num_tokens=128 \
    training.train_temperature=1 \
    .init_from_text \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    .toka \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/projects/bfsg/stalaei/logs/continual_learning/data/cities_easy_synthetic_gen_l8b_non_collapsed_boxed_only_20000_with_subsample_and_original_experiences/dataset_with_up_to_50_new_cities_distractors.jsonl output.local_dir=/projects/bfsg/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/u/stalaei/code/continual-learning/notebooks/experiences.jsonl \
    gen_val_num_repeats=10 .long_seqs dataset.packed_seq_length=82000

torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov6_chatboxed_mixed_dataset \
    kv_cache.num_tokens=128 \
    training.train_temperature=1 \
    .init_from_text \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    .toka \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/projects/bfsg/stalaei/logs/continual_learning/data/cities_easy_synthetic_gen_mixed_15000_boxed_20000_distractors_plus_two_val_datasets/dataset.jsonl \
    output.local_dir=/projects/bfsg/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/u/stalaei/code/continual-learning/notebooks/experiences.jsonl \
    gen_val_num_repeats=10 .long_seqs dataset.packed_seq_length=82000


### brad debugging

python -m src.memory.distillation.distill_into_cartridge \
    run_name=nov6_chatboxed_mixed_dataset \
    kv_cache.num_tokens=128 \
    training.train_temperature=1 \
    .init_from_text \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    .toka \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/projects/bfsg/stalaei/logs/continual_learning/data/cities_easy_synthetic_gen_mixed_15000_boxed_20000_distractors_plus_two_val_datasets/dataset.jsonl \
    output.local_dir=/projects/bfsg/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/u/stalaei/code/continual-learning/notebooks/experiences.jsonl \
    gen_val_num_repeats=10 .long_seqs dataset.packed_seq_length=82000

## Nov 7 (training a cartridge on the second half)

torchrun --nproc_per_node 2 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov7_chatboxed_second_half_cities \
    kv_cache.num_tokens=128 \
    training.train_temperature=1 \
    .init_from_text \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    .toka \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/projects/bfsg/stalaei/logs/continual_learning/data/cities_easy_synthetic_gen_l8b_chatboxed_second_half_seed_23_20000_with_subsample_and_original_experiences/dataset.jsonl \
    output.local_dir=/projects/bfsg/stalaei/continual-learning/cartridges/

## Nov 8 Attention capture
python -m src.attention_capture.run_eval_cli \
    --config /u/stalaei/code/continual-learning/configs/attention_eval/history_agent_cities.yaml \
    --memory-snapshot /projects/bfsg/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/checkpoints/ep_000700/memory_700.jsonl \
    --output-dir /u/stalaei/code/continual-learning/outputs/attn_eval/timing_debug_safe_record \
    --model-type llama \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --layer-idx -1 \
    --query-span-tag current_observation \
    --max-samples 1 \
    --capture-use-eager-attn


## Nov 9, training cartridges with offsets

torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov9_chatboxed_second_half_cities_with_offsets \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=128 \
    non_cartridge_start_position_id_offset=128 \
    training.train_temperature=1 \
    .init_from_text \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    .toka \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/data/stalaei/logs/continual_learning/data/cities_easy_synthetic_gen_l8b_chatboxed_second_half_seed_23_20000_with_subsample_and_original_experiences/dataset.jsonl \
    output.local_dir=/data/stalaei/continual-learning/cartridges/


torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov9_chatboxed_first_half_cities_with_offsets \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    non_cartridge_start_position_id_offset=128 \
    training.train_temperature=1 \
    .init_from_text \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    .toka \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/data/stalaei/logs/continual_learning/data/cities_easy_synthetic_gen_l8b_non_collapsed_boxed_only_20000_with_subsample_and_original_experiences/dataset.jsonl \
    output.local_dir=/data/stalaei/continual-learning/cartridges/


## Nov 10

export CARTRIDGES_DIR=/home/shayant/code/continual-learning/third_party/cartridges
export CARTRIDGES_OUTPUT_DIR=/data/stalaei/capture_attention/
export PYTHONPATH=/home/shayant/code/continual-learning/third_party/cartridges:$PYTHONPATH


### Taking so long
python -m src.attention_capture.run_eval_cli \
    --config configs/attention_eval/history_agent_cities.yaml \
    --memory-snapshot /data/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/checkpoints/ep_000700/memory_700.jsonl \
    --output-dir /data/stalaei/logs/continual_learning/attention_eval/history_agent_cities \
    --model-type llama \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --device cuda \
    --mode generate

### Testing a smaller memory
python -m src.attention_capture.run_eval_cli \
    --config configs/attention_eval/history_agent_cities.yaml \
    --memory-snapshot /data/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_101129/checkpoints/ep_000050/memory_50.jsonl \
    --output-dir /data/stalaei/logs/continual_learning/attention_eval/history_agent_cities \
    --model-type llama \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --device cuda \
    --mode generate

## Nov 11

### small
python -m src.attention_capture.run_eval_cli \
    --config configs/attention_eval/history_agent_cities.yaml \
    --memory-snapshot /data/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_101129/checkpoints/ep_000050/memory_50.jsonl \
    --output-dir /data/stalaei/logs/continual_learning/attention_eval/history_agent_cities_chatboxed_mem_50 \
    --model-type llama \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --layer-idxs 0 16 31

### medium
PROFILE_ATTENTION=1 python -m src.attention_capture.run_eval_cli \
    --config configs/attention_eval/history_agent_cities.yaml \
    --memory-snapshot /data/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_101129/checkpoints/ep_000250/memory_250.jsonl \
    --output-dir /data/stalaei/logs/continual_learning/attention_eval/history_agent_cities_chatboxed_mem_250 \
    --model-type llama \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --layer-idxs 0 16 31 \
    --prompts-idxs 0 20

### large
CUDA_VISIBLE_DEVICES=1 PROFILE_ATTENTION=1 python -m src.attention_capture.run_eval_cli \
    --config configs/attention_eval/history_agent_cities.yaml \
    --memory-snapshot /data/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/checkpoints/ep_000700/memory_700.jsonl \
    --output-dir /data/stalaei/logs/continual_learning/attention_eval/history_agent_cities_chatboxed_mem_700 \
    --model-type llama \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --layer-idxs 0 16 31 \
    --prompts-idxs 0 1 2 3 20 21 22 23



## Nov 11

### cartridge only
PROFILE_ATTENTION=1 python -m src.attention_capture.run_eval_cli \
    --config configs/attention_eval/history_agent_cities.yaml \
    --output-dir /data/stalaei/logs/continual_learning/attention_eval/history_agent_cities_chatboxed_cartridge_nov4_chatboxed \
    --model-type llama \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --layer-idxs 0 16 31 \
    --prompts-idxs 0 1 2 3 20 21 22 23 \
    --cartridge-dir /data/stalaei/continual-learning/cartridges \
    --cartridge-ids nov4_chatboxed

## memory 500
PROFILE_ATTENTION=1 python -m src.attention_capture.run_eval_cli \
    --config configs/attention_eval/history_agent_cities.yaml \
    --memory-snapshot /data/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_101129/checkpoints/ep_000500/memory_500.jsonl \
    --output-dir /data/stalaei/logs/continual_learning/attention_eval/history_agent_cities_chatboxed_mem_500 \
    --model-type llama \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --layer-idxs 0 16 31 \
    --prompts-idxs 0 1 2 3 20 21 22 23

## cartridge + memory 550 (50 experiences only)
CUDA_VISIBLE_DEVICES=2 PROFILE_ATTENTION=1 python -m src.attention_capture.run_eval_cli \
    --config configs/attention_eval/history_agent_cities.yaml \
    --memory-snapshot /data/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_cartridge_nov4_chatboxed-cache-step600/20251105_000849/checkpoints/ep_000550/memory_550.jsonl \
    --output-dir /data/stalaei/logs/continual_learning/attention_eval/history_agent_cities_chatboxed_cartridge_nov4_chatboxed_mem_550 \
    --model-type llama \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --layer-idxs 0 16 31 \
    --prompts-idxs 0 1 2 3 20 21 22 23 \
    --cartridge-dir /data/stalaei/continual-learning/cartridges \
    --cartridge-ids nov4_chatboxed

CUDA_VISIBLE_DEVICES=2 PROFILE_ATTENTION=1 python -m src.attention_capture.run_eval_cli \
    --config configs/attention_eval/history_agent_cities.yaml \
    --memory-snapshot /data/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_cartridge_nov4_chatboxed-cache-step600/20251105_000849/checkpoints/ep_000600/memory_600.jsonl \
    --output-dir /data/stalaei/logs/continual_learning/attention_eval/history_agent_cities_chatboxed_cartridge_nov4_chatboxed_mem_600 \
    --model-type llama \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --layer-idxs 0 16 31 \
    --prompts-idxs 0 1 2 3 20 21 22 23 \
    --cartridge-dir /data/stalaei/continual-learning/cartridges \
    --cartridge-ids nov4_chatboxed

CUDA_VISIBLE_DEVICES=2 PROFILE_ATTENTION=1 python -m src.attention_capture.run_eval_cli \
    --config configs/attention_eval/history_agent_cities.yaml \
    --output-dir /data/stalaei/logs/continual_learning/attention_eval/history_agent_cities_chatboxed_cartridge_nov6_chatboxed_mixed_dataset-cache-step500 \
    --model-type llama \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --layer-idxs 0 16 31 \
    --prompts-idxs 0 1 2 3 20 21 22 23 \
    --cartridge-dir /data/stalaei/continual-learning/cartridges \
    --cartridge-ids nov6_chatboxed_mixed_dataset-cache-step500

CUDA_VISIBLE_DEVICES=2 PROFILE_ATTENTION=1 python -m src.attention_capture.run_eval_cli \
    --config configs/attention_eval/history_agent_cities.yaml \
    --memory-snapshot /data/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_cartridge_nov6_chatboxed_mixed_dataset-cache-step500/20251107_123708/checkpoints/ep_000550/memory_550.jsonl \
    --output-dir /data/stalaei/logs/continual_learning/attention_eval/history_agent_cities_chatboxed_cartridge_nov6_chatboxed_mixed_dataset-cache-step500_mem_550 \
    --model-type llama \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --layer-idxs 0 16 31 \
    --prompts-idxs 0 1 2 3 20 21 22 23 \
    --cartridge-dir /data/stalaei/continual-learning/cartridges \
    --cartridge-ids nov6_chatboxed_mixed_dataset-cache-step500

CUDA_VISIBLE_DEVICES=1 PROFILE_ATTENTION=1 python -m src.attention_capture.run_eval_cli \
    --config configs/attention_eval/history_agent_cities.yaml \
    --memory-snapshot /data/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_cartridge_nov6_chatboxed_mixed_dataset-cache-step500/20251107_123708/checkpoints/ep_000600/memory_600.jsonl \
    --output-dir /data/stalaei/logs/continual_learning/attention_eval/history_agent_cities_chatboxed_cartridge_nov6_chatboxed_mixed_dataset-cache-step500_mem_600 \
    --model-type llama \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --layer-idxs 0 16 31 \
    --prompts-idxs 0 1 2 3 20 21 22 23 \
    --cartridge-dir /data/stalaei/continual-learning/cartridges \
    --cartridge-ids nov6_chatboxed_mixed_dataset-cache-step500


## Training with unrotated queries for cartridges

torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov12_chatboxed_first_half_cities_with_unrot_queries \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/data/stalaei/logs/continual_learning/data/cities_easy_synthetic_gen_l8b_non_collapsed_boxed_only_20000_with_subsample_and_original_experiences/dataset.jsonl \
    output.local_dir=/data/stalaei/continual-learning/cartridges/

torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov12_chatboxed_mixed_dataset_with_unrot_queries \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    .long_seqs \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/data/stalaei/logs/continual_learning/data/cities_easy_synthetic_gen_mixed_15000_boxed_20000_distractors_plus_two_val_datasets/dataset.jsonl \
    output.local_dir=/data/stalaei/continual-learning/cartridges/


## Nov 13
torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov13_chatboxed_mixed_dataset_with_unrot_queries_eval_with_up_to_50_distractors \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    .long_seqs \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/data/stalaei/logs/continual_learning/data/cities_easy_synthetic_gen_mixed_15000_boxed_20000_distractors_plus_two_val_datasets/dataset.jsonl \
    output.local_dir=/data/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/home/shayant/code/continual-learning/notebooks/experiences.jsonl \
    gen_val_num_repeats=10 .long_seqs dataset.packed_seq_length=82000

torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov13_chatboxed_mixed_dataset_with_unrot_queries_eval_with_up_to_50_distractors_1024_tokens_cartridge \
    kv_cache.num_tokens=1024 \
    kv_cache.cartridge_start_position=0 \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    .long_seqs \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/data/stalaei/logs/continual_learning/data/cities_easy_synthetic_gen_mixed_15000_boxed_20000_distractors_plus_two_val_datasets/dataset.jsonl \
    output.local_dir=/data/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/home/shayant/code/continual-learning/notebooks/experiences.jsonl \
    gen_val_num_repeats=10 .long_seqs dataset.packed_seq_length=82000

torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov13_chatboxed_mixed_dataset_with_unrot_queries_eval_with_up_to_50_distractors_8192_tokens_cartridge \
    kv_cache.num_tokens=8192 \
    kv_cache.cartridge_start_position=0 \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    .long_seqs \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/data/stalaei/logs/continual_learning/data/cities_easy_synthetic_gen_mixed_15000_boxed_20000_distractors_plus_two_val_datasets/dataset.jsonl \
    output.local_dir=/data/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/home/shayant/code/continual-learning/notebooks/experiences.jsonl \
    gen_val_num_repeats=10 .long_seqs dataset.packed_seq_length=82000

## Nov 14

torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov14_chatboxed_mixed_dataset_with_rot_queries_eval_with_up_to_50_distractors_8192_tokens_cartridge \
    kv_cache.num_tokens=8192 \
    kv_cache.cartridge_start_position=0 \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=False \
    training.train_temperature=1 \
    .init_from_text \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    .long_seqs \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/data/stalaei/logs/continual_learning/data/cities_easy_synthetic_gen_mixed_15000_boxed_20000_distractors_plus_two_val_datasets/dataset.jsonl \
    output.local_dir=/data/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/home/shayant/code/continual-learning/notebooks/experiences.jsonl \
    gen_val_num_repeats=10 .long_seqs dataset.packed_seq_length=82000

torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov14_chatboxed_mixed_dataset_with_unrot_queries_eval_with_up_to_50_distractors_mlp_residual_8 \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    .long_seqs \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/data/stalaei/logs/continual_learning/data/cities_easy_synthetic_gen_mixed_15000_boxed_20000_distractors_plus_two_val_datasets/dataset.jsonl \
    output.local_dir=/data/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/home/shayant/code/continual-learning/notebooks/experiences.jsonl \
    gen_val_num_repeats=10 .long_seqs dataset.packed_seq_length=82000

torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov14_chatboxed_mixed_dataset_with_unrot_queries_eval_with_up_to_50_distractors_mlp_residual_2 \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=2.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    .long_seqs \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/data/stalaei/logs/continual_learning/data/cities_easy_synthetic_gen_mixed_15000_boxed_20000_distractors_plus_two_val_datasets/dataset.jsonl \
    output.local_dir=/data/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/home/shayant/code/continual-learning/notebooks/experiences.jsonl \
    gen_val_num_repeats=10 .long_seqs dataset.packed_seq_length=82000

## Nov 16
torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov14_chatboxed_mixed_dataset_with_unrot_queries_eval_with_up_to_50_distractors_mlp_residual_2_cartridge_1024 \
    kv_cache.num_tokens=1024 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=2.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    .long_seqs \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/data/stalaei/logs/continual_learning/data/cities_easy_synthetic_gen_mixed_15000_boxed_20000_distractors_plus_two_val_datasets/dataset.jsonl \
    output.local_dir=/data/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/home/shayant/code/continual-learning/notebooks/experiences.jsonl \
    gen_val_num_repeats=10 .long_seqs dataset.packed_seq_length=82000

torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov14_chatboxed_mixed_dataset_with_unrot_queries_eval_with_up_to_50_distractors_mlp_residual_8_cartridge_8192 \
    kv_cache.num_tokens=8192 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    .long_seqs \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/data/stalaei/logs/continual_learning/data/cities_easy_synthetic_gen_mixed_15000_boxed_20000_distractors_plus_two_val_datasets/dataset.jsonl \
    output.local_dir=/data/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/home/shayant/code/continual-learning/notebooks/experiences.jsonl \
    gen_val_num_repeats=10 .long_seqs dataset.packed_seq_length=82000

## Nov 17

torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov17_chatboxed_with_up_to_50_distractors_dataset_with_unrot_queries_mlp_residual_2 \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=2.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    .long_seqs \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/data/stalaei/logs/continual_learning/data/cities_easy_synthetic_gen_l8b_non_collapsed_boxed_only_20000_with_subsample_and_original_experiences/dataset_with_up_to_50_new_cities_distractors.jsonl \
    output.local_dir=/data/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/home/shayant/code/continual-learning/notebooks/experiences.jsonl \
    gen_val_num_repeats=10 .long_seqs dataset.packed_seq_length=64000 \
    generate_batch_size=16 


### Testing the OOM for the above command
torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov17_chatboxed_mixed_dataset_with_unrot_queries_eval_with_up_to_50_distractors_mlp_residual_8 \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    .long_seqs \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/data/stalaei/logs/continual_learning/data/cities_easy_synthetic_gen_mixed_15000_boxed_20000_distractors_plus_two_val_datasets/dataset.jsonl \
    output.local_dir=/data/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/home/shayant/code/continual-learning/notebooks/experiences.jsonl \
    gen_val_num_repeats=10 .long_seqs dataset.packed_seq_length=82000

## Nov 18
torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov18_mixed_dataset_unrot_queries_mlp_residual_8 \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/data/stalaei/logs/continual_learning/data/cities_easy_synthetic_gen_mixed_15000_boxed_20000_distractors_plus_two_val_datasets/dataset.jsonl \
    output.local_dir=/data/stalaei/continual-learning/cartridges/ \
    in_context_examples_path=/data/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_cartridge_nov4_chatboxed-cache-step600/20251105_000849/experiences.jsonl \
    gen_val_num_repeats=1 \
    .long_seqs \
    dataset.packed_seq_length=82000 \
    do_val_gen_eval=F

torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov18_dataset_15000_old_cities_2500_new_cities_unrot_queries_mlp_residual_8 \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/data/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/dataset.jsonl \
    output.local_dir=/data/stalaei/continual-learning/cartridges/ \
    in_context_examples_path=/data/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_cartridge_nov4_chatboxed-cache-step600/20251105_000849/experiences.jsonl \
    gen_val_num_repeats=1 \
    .long_seqs \
    dataset.packed_seq_length=82000

torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov18_dataset_15000_old_cities_2500_new_cities_unrot_queries_mlp_residual_8_toka \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/data/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/dataset.jsonl \
    output.local_dir=/data/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/data/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_cartridge_nov4_chatboxed-cache-step600/20251105_000849/experiences.jsonl \
    gen_val_num_repeats=1 \
    .long_seqs \
    dataset.packed_seq_length=82000

torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov18_mixed_dataset_unrot_queries_mlp_residual_8_toka \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/data/stalaei/logs/continual_learning/data/cities_easy_synthetic_gen_mixed_15000_boxed_20000_distractors_plus_two_val_datasets/dataset.jsonl \
    output.local_dir=/data/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/data/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_cartridge_nov4_chatboxed-cache-step600/20251105_000849/experiences.jsonl \
    gen_val_num_repeats=1 \
    .long_seqs \
    dataset.packed_seq_length=82000

torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov18_dataset_15000_old_cities_2500_new_cities_unrot_queries_mlp_residual_2_toka \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=2.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/data/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/dataset.jsonl \
    output.local_dir=/data/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/data/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_cartridge_nov4_chatboxed-cache-step600/20251105_000849/experiences.jsonl \
    gen_val_num_repeats=1 \
    .long_seqs \
    dataset.packed_seq_length=82000

# Nov 19

## Learning rate sweeps

### node-017
torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov19_dataset_15000_old_cities_2500_new_cities_unrot_queries_mlp_residual_8_toka_lr_e-3 \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=1e-3 \
    input_dataset.local_path=/data/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/dataset.jsonl \
    output.local_dir=/data/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/data/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000

### node-010
torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov19_dataset_15000_old_cities_2500_new_cities_unrot_queries_mlp_residual_8_toka_lr_5e-4 \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-4 \
    input_dataset.local_path=/data/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/dataset.jsonl \
    output.local_dir=/data/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/data/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000

### node-079
torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov19_dataset_15000_old_cities_2500_new_cities_unrot_queries_mlp_residual_8_toka_lr_1e-4 \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=1e-4 \
    input_dataset.local_path=/data/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/dataset.jsonl \
    output.local_dir=/data/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/data/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000

## Across runs variance (15000 old cities + 2500 ICL encouragement dataset)

### node-080
torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov19_dataset_15000_old_cities_2500_new_cities_unrot_queries_mlp_residual_8_toka_lr_5e-3_attempt_1 \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/data/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/dataset.jsonl \
    output.local_dir=/data/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/data/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000

### node-073
torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov19_dataset_15000_old_cities_2500_new_cities_unrot_queries_mlp_residual_8_toka_lr_5e-3_attempt_2 \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/data/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/dataset.jsonl \
    output.local_dir=/data/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/data/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000

### node-078
torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov19_dataset_15000_old_cities_2500_new_cities_unrot_queries_mlp_residual_8_toka_lr_5e-3_attempt_3 \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/data/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/dataset.jsonl \
    output.local_dir=/data/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/data/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000

## Across runs variance (mixed dataset)

### node-076
torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov19_mixed_dataset_unrot_queries_mlp_residual_8_toka_attempt_1 \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/data/stalaei/logs/continual_learning/data/cities_easy_synthetic_gen_mixed_15000_boxed_20000_distractors_plus_two_val_datasets/dataset.jsonl \
    output.local_dir=/data/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/data/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=1 \
    .long_seqs \
    dataset.packed_seq_length=82000

## Dataset with up to 50 first ICL examples + 2500 ICL encouragement

### node-112
torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov19_upto_50_first_ICL_examples_dataset_unrot_queries_mlp_residual_8_toka_attempt_1 \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/data/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_first_distractors_of_the_ICL_run_new_cities_with_1-50_fewshots/dataset.jsonl \
    output.local_dir=/data/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/data/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=1 \
    .long_seqs \
    dataset.packed_seq_length=82000

# Nov 20

### node-076
torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov20_mixed_dataset_unrot_queries_mlp_residual_8_toka_attempt_1 \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/data/stalaei/logs/continual_learning/data/cities_easy_synthetic_gen_mixed_15000_boxed_20000_distractors_plus_two_val_datasets/dataset.jsonl \
    output.local_dir=/data/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/data/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=1 \
    .long_seqs \
    dataset.packed_seq_length=82000

### node-078
torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov20_dataset_15000_old_cities_2500_new_cities_unrot_queries_mlp_residual_8_toka_lr_5e-3_attempt_3 \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/data/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/dataset.jsonl \
    output.local_dir=/data/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/data/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000

### node-112
torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov20_upto_50_first_ICL_examples_dataset_unrot_queries_mlp_residual_8_toka_lr_5e-4 \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-4 \
    input_dataset.local_path=/data/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_first_distractors_of_the_ICL_run_new_cities_with_1-50_fewshots/dataset.jsonl \
    output.local_dir=/data/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/data/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=1 \
    .long_seqs \
    dataset.packed_seq_length=82000

### node-113
torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov20_upto_50_first_ICL_examples_dataset_unrot_queries_mlp_residual_32_toka_lr_5e-4 \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=32.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-4 \
    input_dataset.local_path=/data/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_first_distractors_of_the_ICL_run_new_cities_with_1-50_fewshots/dataset.jsonl \
    output.local_dir=/data/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/data/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=1 \
    .long_seqs \
    dataset.packed_seq_length=82000

### node-041
torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov20_upto_50_first_ICL_examples_dataset_unrot_queries_mlp_residual_8_toka_lr_5e-4_not_shared \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=False \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-4 \
    input_dataset.local_path=/data/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_first_distractors_of_the_ICL_run_new_cities_with_1-50_fewshots/dataset.jsonl \
    output.local_dir=/data/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/data/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=1 \
    .long_seqs \
    dataset.packed_seq_length=82000


# Nov 21

torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov21_dataset_15000_old_cities_2500_new_cities_unrot_queries_mlp_residual_8_toka_lr_5e-3_pos_random \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    kv_cache.positional_embeddings.enabled=True \
    kv_cache.positional_embeddings.init_mode=random \
    kv_cache.positional_embeddings.random_std=0.02 \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/data/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/dataset.jsonl \
    output.local_dir=/data/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/data/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000

# Nov 22

torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov21_dataset_15000_old_cities_2500_new_cities_unrot_queries_mlp_zero_second_layer_residual_8_different_per_layer_lr_5e-3_pos_zero \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    kv_cache.parametrization_zero_init_last_layer=True \
    kv_cache.positional_embeddings.enabled=True \
    kv_cache.positional_embeddings.init_mode=zero \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/data/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/dataset.jsonl \
    output.local_dir=/data/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/data/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000


# Nov 23

### Marlowe n01
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov23_dataset_15000_old_cities_2500_new_cities_unrot_queries_mlp_zero_second_layer_residual_8_different_per_layer_lr_5e-3_pos_zero \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=False \
    kv_cache.parametrization_zero_init_last_layer=True \
    kv_cache.positional_embeddings.enabled=True \
    kv_cache.positional_embeddings.init_mode=zero \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/dataset.jsonl \
    output.local_dir=/scratch/m000122/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/scratch/m000122/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov23_dataset_15000_old_cities_2500_new_cities_unrot_queries_mlp_residual_8_toka_lr_5e-3_pos_random \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    kv_cache.positional_embeddings.enabled=True \
    kv_cache.positional_embeddings.init_mode=random \
    kv_cache.positional_embeddings.random_std=0.02 \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/dataset.jsonl \
    output.local_dir=/scratch/m000122/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/scratch/m000122/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000

### n15
torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov23_dataset_15000_old_cities_2500_new_cities_unrot_queries_mlp_zero_second_layer_residual_8_different_per_layer_lr_5e-3_pos_zero \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=False \
    kv_cache.parametrization_zero_init_last_layer=True \
    kv_cache.positional_embeddings.enabled=True \
    kv_cache.positional_embeddings.init_mode=zero \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/dataset.jsonl \
    output.local_dir=/scratch/m000122/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/scratch/m000122/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000

### n17
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov23_dataset_15000_old_cities_2500_new_cities_unrot_queries_mlp_residual_8_different_per_layer_lr_5e-3_pos_random \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=False \
    kv_cache.positional_embeddings.enabled=True \
    kv_cache.positional_embeddings.init_mode=random \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/dataset.jsonl \
    output.local_dir=/scratch/m000122/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/scratch/m000122/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000

### n15
torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov23_dataset_15000_old_cities_2500_new_cities_unrot_queries_mlp_residual_8_toka_different_per_layer_lr_5e-3_pos_random \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=False \
    kv_cache.positional_embeddings.enabled=True \
    kv_cache.positional_embeddings.init_mode=random \
    kv_cache.positional_embeddings.random_std=0.02 \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/dataset.jsonl \
    output.local_dir=/scratch/m000122/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/scratch/m000122/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000

# Nov 24

### n15 (0,1,2,3)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov24_dataset_15000_old_cities_2500_new_cities_unrot_queries_mlp_residual_8_toka_lr_5e-3_pos_random \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    kv_cache.positional_embeddings.enabled=True \
    kv_cache.positional_embeddings.init_mode=random \
    kv_cache.positional_embeddings.random_std=0.02 \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/dataset.jsonl \
    output.local_dir=/scratch/m000122/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/scratch/m000122/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000

### n15 (4,5,6,7)
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port 29501 --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov24_dataset_15000_old_cities_2500_new_cities_unrot_queries_toka_lr_5e-3 \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    toka_server_port=10301 \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/dataset.jsonl \
    output.local_dir=/scratch/m000122/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/scratch/m000122/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000

### n12 (0,1,2,3)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov24_dataset_15000_old_cities_2500_new_cities_unrot_queries_mlp_zero_second_layer_residual_8_lr_5e-3_pos_zero \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    kv_cache.parametrization_zero_init_last_layer=True \
    kv_cache.positional_embeddings.enabled=True \
    kv_cache.positional_embeddings.init_mode=zero \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/dataset.jsonl \
    output.local_dir=/scratch/m000122/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/scratch/m000122/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000

### n12 (4,5,6,7)
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port 29501 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov24_dataset_15000_old_cities_2500_new_cities_unrot_queries_mlp_zero_second_layer_residual_8_different_per_layer_lr_5e-3_pos_zero \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=False \
    kv_cache.parametrization_zero_init_last_layer=True \
    kv_cache.positional_embeddings.enabled=True \
    kv_cache.positional_embeddings.init_mode=zero \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    toka_server_port=10301 \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/dataset.jsonl \
    output.local_dir=/scratch/m000122/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/scratch/m000122/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000

### n02
torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov24_dataset_15000_old_cities_2500_new_cities_unrot_queries_mlp_zero_second_layer_residual_8_different_per_layer_lr_5e-4_pos_zero \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=False \
    kv_cache.parametrization_zero_init_last_layer=True \
    kv_cache.positional_embeddings.enabled=True \
    kv_cache.positional_embeddings.init_mode=zero \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-4 \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/dataset.jsonl \
    output.local_dir=/scratch/m000122/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/scratch/m000122/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000

### n13
torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov24_dataset_15000_old_cities_2500_new_cities_unrot_queries_mlp_residual_8_toka_lr_5e-3_pos_random \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    kv_cache.positional_embeddings.enabled=True \
    kv_cache.positional_embeddings.init_mode=random \
    kv_cache.positional_embeddings.random_std=0.02 \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/dataset.jsonl \
    output.local_dir=/scratch/m000122/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/scratch/m000122/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000


# Nov 26

### n30
torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov26_dataset_15000_old_cities_2500_new_cities_unrot_queries_mlp_residual_8_toka_lr_5e-3_pos_random \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    kv_cache.positional_embeddings.enabled=True \
    kv_cache.positional_embeddings.init_mode=random \
    kv_cache.positional_embeddings.random_std=0.02 \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/dataset.jsonl \
    output.local_dir=/scratch/m000122/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/scratch/m000122/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000

### gh005
torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=nov26_ataset_150k_old_cities_2500_new_cities_10_splits_unrot_queries_mlp_residual_8_toka_different_per_layer_lr_5e-3_pos_random \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=False \
    kv_cache.positional_embeddings.enabled=True \
    kv_cache.positional_embeddings.init_mode=random \
    kv_cache.positional_embeddings.random_std=0.02 \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/projects/bfsg/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/dataset_150k_old_cities_2500_new_cities_10_splits.jsonl \
    output.local_dir=/projects/bfsg/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/projects/bfsg/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000

# Dec 1

### n05 (0,1,2,3)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=dec1_dataset_15000_old_cities_2500_new_cities_unrot_queries_mlp_residual_8_toka_lr_5e-3_pos_random \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    kv_cache.positional_embeddings.enabled=True \
    kv_cache.positional_embeddings.init_mode=random \
    kv_cache.positional_embeddings.random_std=0.02 \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/dataset.jsonl \
    output.local_dir=/scratch/m000122/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/scratch/m000122/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000

### n05 (4,5,6,7)
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port 29501 -m src.memory.distillation.distill_into_cartridge \
    run_name=dec1_dataset_15000_old_cities_2500_new_cities_unrot_queries_mlp_residual_8_per_layer_toka_lr_5e-3_pos_random \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=False \
    kv_cache.positional_embeddings.enabled=True \
    kv_cache.positional_embeddings.init_mode=random \
    kv_cache.positional_embeddings.random_std=0.02 \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    toka_server_port=10301 \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/dataset.jsonl \
    output.local_dir=/scratch/m000122/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/scratch/m000122/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000

### n04 (0,1,2,3)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=dec1_dataset_150k_old_cities_2500_new_cities_10_splits_unrot_queries_mlp_residual_8_toka_different_per_layer_lr_5e-3_pos_random \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    kv_cache.positional_embeddings.enabled=True \
    kv_cache.positional_embeddings.init_mode=random \
    kv_cache.positional_embeddings.random_std=0.02 \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/dataset_150k_old_cities_2500_new_cities_10_splits.jsonl \
    output.local_dir=/scratch/m000122/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/scratch/m000122/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000

### n04 (4,5,6,7)
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port 29501 -m src.memory.distillation.distill_into_cartridge \
    run_name=dec1_dataset_150k_old_cities_2500_new_cities_10_splits_unrot_queries_mlp_residual_8_per_layer_toka_lr_5e-3_pos_random \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=False \
    kv_cache.positional_embeddings.enabled=True \
    kv_cache.positional_embeddings.init_mode=random \
    kv_cache.positional_embeddings.random_std=0.02 \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    toka_server_port=10301 \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/dataset_150k_old_cities_2500_new_cities_10_splits.jsonl \
    output.local_dir=/scratch/m000122/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/scratch/m000122/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000

# Dec 2

### n12 (0,1,2,3)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=dec2_dataset_150k_old_cities_2500_new_cities_10_splits_unrot_queries_mlp_residual_8_toka_different_per_layer_lr_5e-4_pos_random \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    kv_cache.positional_embeddings.enabled=True \
    kv_cache.positional_embeddings.init_mode=random \
    kv_cache.positional_embeddings.random_std=0.02 \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-4 \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/dataset_150k_old_cities_2500_new_cities_10_splits.jsonl \
    output.local_dir=/scratch/m000122/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/scratch/m000122/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000

### n12 (4,5,6,7)
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port 29501 -m src.memory.distillation.distill_into_cartridge \
    run_name=dec2_dataset_150k_old_cities_2500_new_cities_10_splits_unrot_queries_mlp_residual_8_per_layer_toka_lr_5e-4_pos_random \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=False \
    kv_cache.positional_embeddings.enabled=True \
    kv_cache.positional_embeddings.init_mode=random \
    kv_cache.positional_embeddings.random_std=0.02 \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    toka_server_port=10301 \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-4 \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/dataset_150k_old_cities_2500_new_cities_10_splits.jsonl \
    output.local_dir=/scratch/m000122/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/scratch/m000122/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000

# Dec 6
### n17 (0,1,2,3)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=dec6_dataset_20k_old_cities_2500_unrot_queries_mlp_residual_8_toka_different_per_layer_lr_5e-3_pos_random \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=False \
    kv_cache.positional_embeddings.enabled=True \
    kv_cache.positional_embeddings.init_mode=random \
    kv_cache.positional_embeddings.random_std=0.02 \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/cities_easy_synthetic_gen_l8b_non_collapsed_boxed_only_20000_with_subsample_and_original_experiences/dataset_with_up_to_50_new_cities_distractors.jsonl \
    output.local_dir=/scratch/m000122/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/scratch/m000122/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000

### n17 (4,5,6,7)
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port 29501 -m src.memory.distillation.distill_into_cartridge \
    run_name=dec6_dataset_150k_old_cities_2500_new_cities_10_splits_unrot_queries_1024_mlp_residual_8_different_per_layer_toka_lr_5e-4_pos_random_std_02 \
    kv_cache.num_tokens=1024 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=False \
    kv_cache.positional_embeddings.enabled=True \
    kv_cache.positional_embeddings.init_mode=random \
    kv_cache.positional_embeddings.random_std=0.02 \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    toka_server_port=10301 \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-4 \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/dataset_150k_old_cities_2500_new_cities_10_splits.jsonl \
    output.local_dir=/scratch/m000122/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/scratch/m000122/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000

### n18 (0,1,2,3)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=dec6_dataset_12.5_2.5_2.5_unrot_queries_mlp_residual_8_toka_different_per_layer_lr_5e-3_pos_random \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=False \
    kv_cache.positional_embeddings.enabled=True \
    kv_cache.positional_embeddings.init_mode=random \
    kv_cache.positional_embeddings.random_std=0.02 \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/2500_old_no_dist_12500_old_upto_50_dist_2500_new_ICL.jsonl \
    output.local_dir=/scratch/m000122/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/scratch/m000122/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000

### n18 (4,5,6,7)
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port 29501 -m src.memory.distillation.distill_into_cartridge \
    run_name=dec6_dataset_12.5_2.5_2.5_unrot_queries_mlp_residual_8_toka_different_per_layer_lr_5e-4_pos_random \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=False \
    kv_cache.positional_embeddings.enabled=True \
    kv_cache.positional_embeddings.init_mode=random \
    kv_cache.positional_embeddings.random_std=0.02 \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    toka_server_port=10301 \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-4 \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/2500_old_no_dist_12500_old_upto_50_dist_2500_new_ICL.jsonl \
    output.local_dir=/scratch/m000122/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/scratch/m000122/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000

### n19 (0,1,2,3)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=dec6_dataset_20k_old_cities_2500_unrot_queries_mlp_residual_8_toka_different_per_layer_lr_5e-4_pos_random \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=False \
    kv_cache.positional_embeddings.enabled=True \
    kv_cache.positional_embeddings.init_mode=random \
    kv_cache.positional_embeddings.random_std=0.02 \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-4 \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/cities_easy_synthetic_gen_l8b_non_collapsed_boxed_only_20000_with_subsample_and_original_experiences/dataset_with_up_to_50_new_cities_distractors.jsonl \
    output.local_dir=/scratch/m000122/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/scratch/m000122/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000

### n19 (4,5,6,7)
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port 29501 -m src.memory.distillation.distill_into_cartridge \
    run_name=dec6_dataset_12.5_2.5_2.5_unrot_queries_1024_mlp_residual_8_toka_different_per_layer_lr_5e-4_pos_random \
    kv_cache.num_tokens=1024 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=False \
    kv_cache.positional_embeddings.enabled=True \
    kv_cache.positional_embeddings.init_mode=random \
    kv_cache.positional_embeddings.random_std=0.02 \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    toka_server_port=10301 \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-4 \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/2500_old_no_dist_12500_old_upto_50_dist_2500_new_ICL.jsonl \
    output.local_dir=/scratch/m000122/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/scratch/m000122/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000

# Dec 8
### n18 
torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=dec8_dataset_15000_old_cities_2500_new_cities_unrot_queries_mlp_residual_8_toka_lr_5e-3_pos_random \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    kv_cache.positional_embeddings.enabled=True \
    kv_cache.positional_embeddings.init_mode=random \
    kv_cache.positional_embeddings.random_std=0.02 \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    training.save_every_n_steps=50 \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/dataset.jsonl \
    output.local_dir=/scratch/m000122/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/scratch/m000122/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000

### n05
torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=dec8_dataset_12.5_2.5_2.5_unrot_queries_mlp_residual_8_toka_lr_5e-3_pos_random \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    kv_cache.positional_embeddings.enabled=True \
    kv_cache.positional_embeddings.init_mode=random \
    kv_cache.positional_embeddings.random_std=0.02 \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    training.save_every_n_steps=50 \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/2500_old_no_dist_12500_old_upto_50_dist_2500_new_ICL.jsonl \
    output.local_dir=/scratch/m000122/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/scratch/m000122/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000

### n04 (separate_sum attention)
torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=dec8_dataset_15000_old_cities_2500_new_cities_unrot_queries_mlp_residual_8_toka_lr_5e-3_pos_random_sep_sum \
    kv_cache.num_tokens=128 \
    kv_cache.cartridge_start_position=0 \
    kv_cache.parametrization_type=mlp_residual \
    kv_cache.parametrization_hidden_multiplier=8.0 \
    kv_cache.parametrization_activation=relu \
    kv_cache.parametrization_share_across_layers=True \
    kv_cache.positional_embeddings.enabled=True \
    kv_cache.positional_embeddings.init_mode=random \
    kv_cache.positional_embeddings.random_std=0.02 \
    non_cartridge_start_position_id_offset=0 \
    use_unrotated_queries_for_cartridges=True \
    cartridge_attention_mode=separate_sum \
    training.train_temperature=1 \
    .init_from_text \
    .toka \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/cities_easy/brad_magic_on_top_shayan_finesse.txt \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .train_gen_eval \
    .synth_cities \
    training.weight_decay=0.0 \
    training.lr=5e-3 \
    training.save_every_n_steps=50 \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/cities_easy_synthetic_old_cities_with_up_to_50_distractors_new_cities_with_1-50_fewshots/dataset.jsonl \
    output.local_dir=/scratch/m000122/stalaei/continual-learning/cartridges/ \
    gen_max_incontext_examples=50 \
    gen_min_incontext_examples=0 \
    in_context_examples_path=/scratch/m000122/stalaei/logs/continual_learning/outputs/stalaei_cities_easy/history_agent/easy_synth_cities_40_25_l8b_non_collapsed_only_boxed/20251103_181024/experiences.jsonl \
    gen_val_num_repeats=5 \
    .long_seqs \
    dataset.packed_seq_length=82000