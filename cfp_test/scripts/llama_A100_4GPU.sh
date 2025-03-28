#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
DEVICE_NUM=4
DATA_TYPE="float32"

TEST_CONFIG=(
    "4096, 32, 4"
    "4096, 64, 4"
    "4096, 80, 4"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="$(cd "$SCRIPT_DIR/../llama2" && pwd)" 

for CONFIG_LINE in "${TEST_CONFIG[@]}"; do
    IFS=',' read -r CONFIG BATCH_SIZE NUM_LAYER <<< "$CONFIG_LINE"
    $LLAMA_DIR/llama_test.sh \
        --config "$CONFIG" \
        --batch_size "$BATCH_SIZE" \
        --num_layer "$NUM_LAYER" \
        --device_num "$DEVICE_NUM" \
        --data_type "$DATA_TYPE" \
        --parallel_block_opt 1 

    echo "================================================================="
done
unset CUDA_VISIBLE_DEVICES