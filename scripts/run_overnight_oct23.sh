torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=oct23_250train_128tokens_sysmem_wd01 \
    kv_cache.num_tokens=128 \
    training.train_temperature=1 \
    .init_from_text \
    kv_cache.init_text_file=src/memory/distillation/kv_cache_init_texts/v1.txt \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/finer_v1_train_ICL_exclude_current_250_triplets_false_1000_reps_temp_0.7/dataset.jsonl \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/finer/system_prompt_brad_magic.txt \
    training.lr=5e-4 \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .toka \
    .train_gen_eval \
    training.weight_decay=0.1 \
    config.epochs=1

torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=oct23_250train_128tokens_sysmem_wd1e3 \
    kv_cache.num_tokens=128 \
    training.train_temperature=1 \
    .init_from_text \
    kv_cache.init_text_file=src/memory/distillation/kv_cache_init_texts/v1.txt \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/finer_v1_train_ICL_exclude_current_250_triplets_false_1000_reps_temp_0.7/dataset.jsonl \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/finer/system_prompt_brad_magic.txt \
    training.lr=5e-4 \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .toka \
    .train_gen_eval \
    training.weight_decay=1e-3 \
    config.epochs=1

torchrun --nproc_per_node 4 -m src.memory.distillation.distill_into_cartridge \
    run_name=oct23_250train_128tokens_sysmem_wd1 \
    kv_cache.num_tokens=128 \
    training.train_temperature=1 \
    .init_from_text \
    kv_cache.init_text_file=src/memory/distillation/kv_cache_init_texts/v1.txt \
    input_dataset.local_path=/scratch/m000122/stalaei/logs/continual_learning/data/finer_v1_train_ICL_exclude_current_250_triplets_false_1000_reps_temp_0.7/dataset.jsonl \
    do_loss_evals=F \
    system_prompt_path=src/data/prompts/finer/system_prompt_brad_magic.txt \
    training.lr=5e-4 \
    generate_eval_every_n_steps=50 \
    streaming_dataset=T \
    dataloader_num_workers=8 \
    .streaming \
    .toka \
    .train_gen_eval \
    training.weight_decay=1 \
    config.epochs=1