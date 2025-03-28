#!/bin/bash

DEVICE_NUM=8
DATA_TYPE="float32"

TEST_CONFIG=(
    "6656, 16, 2"
    "6656, 64, 2"
    "6656, 96, 2"
    "6656, 128, 2"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="$(cd "$SCRIPT_DIR/../llama2" && pwd)" 

for CONFIG_LINE in "${TEST_CONFIG[@]}"; do
    IFS=',' read -r CONFIG BATCH_SIZE NUM_LAYER <<< "$CONFIG_LINE"
    $LLAMA_DIR/llama2_test.sh \
        --d_model "$CONFIG" \
        --batch_size "$BATCH_SIZE" \
        --num_layer "$NUM_LAYER" \
        --device_num "$DEVICE_NUM" \
        --data_type "$DATA_TYPE" \

    echo "================================================================="
done