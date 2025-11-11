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