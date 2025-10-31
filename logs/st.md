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


## Data Generation

python -m src.memory.distillation.