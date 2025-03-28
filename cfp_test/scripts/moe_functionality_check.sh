#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
DEVICE_NUM=4
DATA_TYPE="float32"

TEST_CONFIG=(
    "E, 32, 4"
    "E, 64, 4"
    "E, 96, 2"
    "E, 128, 2"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOE_DIR="$(cd "$SCRIPT_DIR/../moe" && pwd)" 


for CONFIG_LINE in "${TEST_CONFIG[@]}"; do
    IFS=',' read -r CONFIG BATCH_SIZE NUM_LAYER <<< "$CONFIG_LINE"
    $MOE_DIR/moe_test.sh \
        --config "$CONFIG" \
        --batch_size "$BATCH_SIZE" \
        --num_layer "$NUM_LAYER" \
        --device_num "$DEVICE_NUM" \
        --data_type "$DATA_TYPE" \
        --parallel_block_opt 1 

    echo "================================================================="
done
unset CUDA_VISIBLE_DEVICES