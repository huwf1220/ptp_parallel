#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
CONFIG="C"
BATCH_SIZE=8
NUM_LAYER=2
DEVICE_NUM=4
DATA_TYPE="float32"
PARALLEL_BLOCK_OPT=1 
DEBUG_DIR="tmp"
DUMP_COMM_VOLUME="false"
ENUM_AND_BENCH=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --num_layer) NUM_LAYER="$2"; shift ;;
        --device_num) DEVICE_NUM="$2"; shift ;;
        --data_type) DATA_TYPE="$2"; shift ;;
        --parallel_block_opt) PARALLEL_BLOCK_OPT="$2"; shift ;;
        --debug_dir) DEBUG_DIR="$2"; shift ;;
        --dump_comm_volume) DUMP_COMM_VOLUME="$2"; shift ;;
        --enum_and_bench) ENUM_AND_BENCH="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [[ "$PARALLEL_BLOCK_OPT" -eq 1 ]]; then
    PARALLEL_BLOCK_OPT_FLAG="--parallel_block_opt"
else
    PARALLEL_BLOCK_OPT_FLAG=""
fi

if [[ "$ENUM_AND_BENCH" -eq 1 ]]; then
    ENUM_AND_BENCH_FLAG="--enum_and_bench"
else
    ENUM_AND_BENCH_FLAG=""
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BERT_DIR="$(cd "$SCRIPT_DIR/../../bert" && pwd)" 

python $BERT_DIR/bert_benchmark_combination.py \
    --config "$CONFIG" \
    --batch_size "$BATCH_SIZE" \
    --num_layer "$NUM_LAYER" \
    --device_num "$DEVICE_NUM" \
    --data_type "$DATA_TYPE" \
    $PARALLEL_BLOCK_OPT_FLAG \
    --debug_dir "$DEBUG_DIR" \
    --dump_comm_volume "$DUMP_COMM_VOLUME" \
    $ENUM_AND_BENCH_FLAG

unset CUDA_VISIBLE_DEVICES