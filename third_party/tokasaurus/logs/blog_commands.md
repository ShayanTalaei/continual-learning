Setup:

```bash

BASE_DIR=local/results
TOKA_ENV=toka-bench
VLLM_ENV=vllm-bench
SGL_ENV=sgl-bench

conda create -n $TOKA_ENV -y python=3.12
conda activate $TOKA_ENV
conda install -y nvidia/label/cuda-12.4.1::cuda-toolkit
cd ~/tokasaurus
pip install uv
uv pip install -e '.[dev]'

conda create -n $VLLM_ENV -y python=3.12
conda activate $VLLM_ENV
conda install -y nvidia/label/cuda-12.4.1::cuda-toolkit
pip install uv
uv pip install vllm==0.9.0.1
uv pip install flashinfer-python==0.2.5 --no-deps

conda create -n $SGL_ENV -y python=3.12
conda activate $SGL_ENV
conda install -y nvidia/label/cuda-12.4.1::cuda-toolkit
pip install uv
uv pip install 'sglang[all]'==0.4.6.post5

```


1xH100:

```bash

# monkeys 1b 1xh100

DIR=$BASE_DIR/monkeys-1b-1xh100

TOKA_COMMAND="tksrs model=meta-llama/Llama-3.2-1B-Instruct kv_cache_num_tokens='((1024 + 512) * 1024)' max_seqs_per_forward=8192 max_tokens_per_forward=32768 torch_compile=T use_hydragen=T hydragen_min_group_size=129 .uvsh"

TOKA_NO_HYDRAGEN_COMMAND="tksrs model=meta-llama/Llama-3.2-1B-Instruct kv_cache_num_tokens='((1024 + 512) * 1024)' max_seqs_per_forward=8192 max_tokens_per_forward=32768 torch_compile=T .uvsh"

SGL_COMMAND="python -m sglang.launch_server --model-path meta-llama/Llama-3.2-1B-Instruct --port 10210 --max-running-requests 8192 --chunked-prefill-size 32768 --max-prefill-tokens 32786 --max-total-tokens 1572864 --schedule-conservativeness 0.01 --log-level-http warning --attention-backend flashinfer"

VLLM_COMMAND="vllm serve meta-llama/Llama-3.2-1B-Instruct --num-gpu-blocks-override 98304 --port 10210 --max-num-seqs 8192 --max-num-batched-tokens 32768 --enable-prefix-caching --disable-log-requests --enable-chunked-prefill --uvicorn-log-level warning"

bench() {
    python tokasaurus/benchmarks/monkeys_gsm8k.py model=meta-llama/Llama-3.2-1B-Instruct limit=128 n=1024 port=10210 reps=4 "$@"
}

ulimit -n unlimited
export SGLANG_DETOKENIZER_MAX_STATES=10000000 

conda activate $TOKA_ENV
bench launch=$TOKA_COMMAND save_path=$DIR/toka.jsonl
bench launch=$TOKA_NO_HYDRAGEN_COMMAND save_path=$DIR/toka_no_hydragen.jsonl
bench launch=$SGL_COMMAND env=$SGL_ENV save_path=$DIR/sgl.jsonl
bench launch=$VLLM_COMMAND env=$VLLM_ENV save_path=$DIR/vllm.jsonl

# sharegpt 1b 1xh100

DIR=$BASE_DIR/sharegpt-1b-1xh100

BENCH_COMMAND='python3 -m sglang.bench_serving --backend vllm --dataset-name sharegpt --num-prompt 65536 --sharegpt-context-len 131072 --model meta-llama/Llama-3.2-1B-Instruct --disable-stream --max-concurrency 8192 --port 10210'

TOKA_COMMAND="tksrs model=meta-llama/Llama-3.2-1B-Instruct kv_cache_num_tokens='((1024 + 768) * 1024)' max_seqs_per_forward=8192 max_tokens_per_forward=32768 torch_compile=T scheduling_steps_ahead=16 .uvsh"

SGL_COMMAND="python -m sglang.launch_server --model-path meta-llama/Llama-3.2-1B-Instruct --port 10210 --max-running-requests 8192 --chunked-prefill-size 32768 --max-prefill-tokens 32786 --max-total-tokens 1835008 --schedule-conservativeness 0.1 --log-level-http warning --attention-backend flashinfer"

VLLM_COMMAND="vllm serve meta-llama/Llama-3.2-1B-Instruct --num-gpu-blocks-override 114688 --port 10210 --max-num-seqs 8192 --max-num-batched-tokens 32768 --enable-prefix-caching --disable-log-requests --enable-chunked-prefill --uvicorn-log-level warning"

bench() {
    python tokasaurus/benchmarks/sharegpt.py model=meta-llama/Llama-3.2-1B-Instruct sharegpt_command=\"$BENCH_COMMAND\" sharegpt_env=$SGL_ENV port=10210 reps=4 "$@"
}

ulimit -n unlimited
export SGLANG_DETOKENIZER_MAX_STATES=10000000

bench launch=$TOKA_COMMAND save_path=$DIR/toka.jsonl
bench launch=$SGL_COMMAND env=$SGL_ENV save_path=$DIR/sgl.jsonl
bench launch=$VLLM_COMMAND env=$VLLM_ENV save_path=$DIR/vllm.jsonl

```


8xH100:

```bash

# sharegpt 70b 8xh100

DIR=$BASE_DIR/sharegpt-70b-8xh100

BENCH_COMMAND='python3 -m sglang.bench_serving --backend vllm --dataset-name sharegpt --num-prompt 65536 --sharegpt-context-len 131072 --model meta-llama/Llama-3.1-70B-Instruct --disable-stream --max-concurrency 8192 --port 10210'

TOKA_COMMAND="tksrs model=meta-llama/Llama-3.1-70B-Instruct tp_size=8 kv_cache_num_tokens='((1024 + 128) * 1024)' max_seqs_per_forward=4096 max_tokens_per_forward=16384 torch_compile=T async_tp_threshold=6144 .uvsh"

SGL_COMMAND="python -m sglang.launch_server --model-path meta-llama/Llama-3.1-70B-Instruct --port 10210 --max-running-requests 4096 --chunked-prefill-size 49152 --max-prefill-tokens 49152 --max-total-tokens 1179648 --schedule-conservativeness 0.1 --tensor-parallel-size 8 --log-level-http warning --attention-backend flashinfer"

VLLM_COMMAND="vllm serve meta-llama/Llama-3.1-70B-Instruct --num-gpu-blocks-override 73728 --port 10210 --max-num-seqs 4096 --max-num-batched-tokens 16384 --enable-prefix-caching --disable-log-requests --enable-chunked-prefill --tensor-parallel-size 8 --uvicorn-log-level warning"

bench() {
    python tokasaurus/benchmarks/sharegpt.py model=meta-llama/Llama-3.1-70B-Instruct sharegpt_command=\"$BENCH_COMMAND\" sharegpt_env=$SGL_ENV port=10210 reps=4 "$@"
}

ulimit -n unlimited
export SGLANG_DETOKENIZER_MAX_STATES=10000000

conda activate $TOKA_ENV

bench launch=$TOKA_COMMAND save_path=$DIR/toka.jsonl
bench launch=$SGL_COMMAND env=$SGL_ENV save_path=$DIR/sgl.jsonl
bench launch=$VLLM_COMMAND env=$VLLM_ENV save_path=$DIR/vllm.jsonl


```


8xL40S:

```bash

DIR=$BASE_DIR/sharegpt-70b-8xl40s

BENCH_COMMAND='python3 -m sglang.bench_serving --backend vllm --dataset-name sharegpt --num-prompt 16384 --sharegpt-context-len 4096 --model meta-llama/Llama-3.1-70B-Instruct --disable-stream --max-concurrency 4096 --port 10210'

TOKA_COMMAND="tksrs model=meta-llama/Llama-3.1-70B-Instruct pp_size=8 kv_cache_num_tokens='(512 * 1024)' max_seqs_per_forward=2048 max_tokens_per_forward=8192 torch_compile=T .uvsh"

VLLM_COMMAND="vllm serve meta-llama/Llama-3.1-70B-Instruct --num-gpu-blocks-override 32768 --port 10210 --max-num-seqs 512 --max-num-batched-tokens 2048 --enable-prefix-caching --disable-log-requests --enable-chunked-prefill --pipeline-parallel-size 8 --port 10210 --max-model-len 32768"

SGL_COMMAND="python -m sglang.launch_server --model-path meta-llama/Llama-3.1-70B-Instruct --port 10210 --max-running-requests 2048 --chunked-prefill-size 8192 --max-prefill-tokens 8192 --max-total-tokens 524288 --schedule-conservativeness 0.1 --pipeline-parallel-size 8 --log-level-http warning --attention-backend flashinfer"

bench() {
    python tokasaurus/benchmarks/sharegpt.py model=meta-llama/Llama-3.1-70B-Instruct sharegpt_command=\"$BENCH_COMMAND\" sharegpt_env=$SGL_ENV port=10210 reps=4 "$@"
}

ulimit -n unlimited
export SGLANG_DETOKENIZER_MAX_STATES=10000000

conda activate $TOKA_ENV

bench launch=$TOKA_COMMAND save_path=$DIR/toka.jsonl
bench launch=$SGL_COMMAND env=$SGL_ENV save_path=$DIR/sgl.jsonl
bench launch=$VLLM_COMMAND env=$VLLM_ENV save_path=$DIR/vllm.jsonl


```