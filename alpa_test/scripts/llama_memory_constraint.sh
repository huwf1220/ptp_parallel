#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
D_MODEL=4096
DEVICE_NUM=4
DATA_TYPE="float32"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="$(cd "$SCRIPT_DIR/../llama2" && pwd)" 

NUM_LAYER=6
BATCH_SIZES=(16 32 48 64 80 96 128 144 160)
for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    $LLAMA_DIR/llama2_test.sh \
        --d_model "$D_MODEL" \
        --batch_size "$BATCH_SIZE" \
        --num_layer "$NUM_LAYER" \
        --device_num "$DEVICE_NUM" \
        --data_type "$DATA_TYPE" 
    echo "====================================================================================================================="
done

BATCH_SIZE=128
NUM_LAYERS=(2 3 4 5 6 7 8)

for NUM_LAYER in "${NUM_LAYERS[@]}"; do
    $LLAMA_DIR/llama2_test.sh \
        --d_model "$D_MODEL" \
        --batch_size "$BATCH_SIZE" \ 
        --num_layer "$NUM_LAYER" \
        --device_num "$DEVICE_NUM" \
        --data_type "$DATA_TYPE" 
    echo "====================================================================================================================="
done
unset CUDA_VISIBLE_DEVICES