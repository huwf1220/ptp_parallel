#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
DEVICE_NUM=4
DATA_TYPE="float16"

TEST_CONFIG=(
    "E, 16, 4"
    "E, 32, 2"
    "E, 64, 2"
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

    echo "================================================================="
done
unset CUDA_VISIBLE_DEVICES